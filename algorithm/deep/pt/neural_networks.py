import torch
import torch.nn as nn
import torch.nn.functional as F

from ..exceptions import NotFoundOptimizerException, NotImplementedOptimizerException
from ..utils.pt import is_nevergrad_optimizer, is_torch_optimizer
from .adr import ADRParameterSpace, Branch, build_dr_layers




def _channels_first(input_shape):
    if len(input_shape) != 3:
        raise ValueError("input_shape must have 3 entries")
    if input_shape[0] not in (1, 3, 4) and input_shape[-1] in (1, 3, 4):
        h, w, c = input_shape
        return c, h, w
    return tuple(input_shape)



class ADRModel(nn.Module):

    def __init__(self, input_shape=(3, 128, 128), field='data',
                 domain_randomization=None, output_activation=None, verbose=True):
        super().__init__()
        self.input_shape_ = _channels_first(input_shape)
        self.field = field
        self.output_activation = output_activation

        self.model_branch1 = Branch(name="Main Branch", module=self)
        self.domain_randomization = False
        self.optimize = False
        self.domain_randomization__mode = None
        self.parameter_space = None
        self.model_branch2 = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics_fns = {}
        self.fverbose = 0
        self.fverbose_path = None
        self.fverbose_file = None
        self._dr_generator = None

        if domain_randomization is not None:
            self._setup_domain_randomization(domain_randomization)

        if verbose:
            if self.optimize:
                print("-> 2 optimizers needed when calling 'model.compile'\n")
            print("'model.compile' parameters info:")
            print("\tclass name required for loss and optimizers (e.g.: "
                  "torch.nn.CrossEntropyLoss or torch.optim.Adam or ng.optimizers.CMA)")
            print("\tany additional parameters required must be passed as a dictionary in "
                  "the second element of the tuple (class_name, parameters) "
                  "(e.g.: (torch.optim.Adam, {'lr': 1e-3})")


    def _setup_domain_randomization(self, domain_randomization):
        self.domain_randomization = True
        self.domain_randomization__mode = domain_randomization.mode
        self.optimize = bool(domain_randomization.optimize)
        parameters_name = domain_randomization.get_parameters_list()
        self.normalized_space = {"lower": 0.0, "upper": 4.0}

        if domain_randomization.seed is not None:
            self._dr_generator = torch.Generator().manual_seed(int(domain_randomization.seed))

        if self.optimize:
            self.model_branch2 = Branch(name="Random Distribution parameters branch")
            self.parameter_space = ADRParameterSpace(
                domain_randomization,
                n_parameters=len(parameters_name),
                normalized_space=self.normalized_space,
            )
            self.model_branch2.add_variables(self.parameter_space.variable_names())

        layers = build_dr_layers(domain_randomization, parameters_name, self.optimize)
        if "parameters" in layers:
            self.branch1_random_parameters = layers["parameters"]
        else:
            self.branch1_random_brightness = layers["brightness"]
            self.branch1_random_contrast = layers["contrast"]
            self.branch1_random_horizontally_flip = layers["horizontally flip"]
            self.branch1_random_vertically_flip = layers["vertically flip"]
            self.branch1_random_hue = layers["hue"]
            self.branch1_random_jpeg_quality = layers["jpeg quality"]
            self.branch1_random_saturation = layers["saturation"]

    _TRANSFORM_ATTRS = ("brightness", "contrast", "horizontally_flip", "vertically_flip",
                        "hue", "jpeg_quality", "saturation")

    def _randomize(self, data, training):
        if not self.domain_randomization:
            return data
        joint = getattr(self, "branch1_random_parameters", None)

        if self.optimize and training:
            n = data.shape[0] if data.ndim == 4 else 1
            sampled = self.parameter_space.sample(n, generator=self._dr_generator)
            if joint is not None:
                return joint(data, values=sampled, rand=False, training=training)
            for value, name in zip(sampled, self._TRANSFORM_ATTRS):
                data = getattr(self, f"branch1_random_{name}")(
                    data, value=value, rand=False, training=training)
            return data

        if joint is not None:
            return joint(data, training=training)
        for name in self._TRANSFORM_ATTRS:
            data = getattr(self, f"branch1_random_{name}")(data, training=training)
        return data


    def _features(self, x):
        raise NotImplementedError

    def forward(self, inputs, training=None):
        if training is None:
            training = self.training
        data = inputs.get(self.field, inputs) if isinstance(inputs, dict) else inputs
        data = self._randomize(data, training)
        y = self._features(data)
        if self.output_activation == "softmax":
            y = torch.softmax(y, dim=-1)
        return y

    def _flatten_features(self, extractor):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            probe = torch.zeros(1, *self.input_shape_)
            n = extractor(probe).flatten(1).shape[1]
        self.train(was_training)
        return n


    def compile(self, optimizer, loss=None, metrics=None):
        par = None
        try:
            if isinstance(optimizer[0], tuple):
                opt, par = optimizer[0]
            else:
                opt = optimizer[0]
        except TypeError:
            if isinstance(optimizer, tuple):
                opt, par = optimizer
            else:
                opt = optimizer

        par = dict(par) if par is not None else {}
        trainable = [p for p in self.parameters() if p.requires_grad]
        self.model_branch1.optimizer = opt(trainable, **par)

        if self.domain_randomization and self.optimize:
            par = None
            if isinstance(optimizer[1], tuple):
                opt, par = optimizer[1]
            else:
                opt = optimizer[1]
            par = dict(par) if par is not None else {}
            par["parametrization"] = self.parameter_space.parametrization()
            self.model_branch2.optimizer = opt(**par)

        if self.model_branch1.optimizer is None or (
                self.domain_randomization and self.optimize
                and self.model_branch2.optimizer is None):
            raise NotFoundOptimizerException()

        if loss is not None:
            self.loss_fn = loss
        self.metrics_fns = metrics or {}
        return self


    def train_step(self, data):
        imgs, labs = data
        self.train()

        predictions = self(imgs, training=True)
        loss = self.loss_fn(predictions, labs)
        loss_branch2 = loss.detach()
        if self.domain_randomization and self.optimize:
            loss_branch2 = loss_branch2 + self.parameter_space.penalty()

        self._update_parameters(self.model_branch1.optimizer, loss)
        if self.domain_randomization and self.optimize:
            self._update_parameters(self.model_branch2.optimizer, loss_branch2)

        logs = {"loss": float(loss.detach())}
        for name, fn in self.metrics_fns.items():
            logs[name] = float(fn(predictions.detach(), labs))
        return logs

    def _update_parameters(self, optimizer, loss=None):
        if is_nevergrad_optimizer(optimizer):
            if loss is not None:
                self.parameter_space.tell(optimizer, loss)
            self.parameter_space.ask(optimizer)
            if self.fverbose in (2, 3):
                self._print_fverbose(branch=2)
        elif is_torch_optimizer(optimizer):
            if loss is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            if self.fverbose in (1, 3):
                self._print_fverbose(branch=1)
        else:
            raise NotImplementedOptimizerException(optimizer)

    def fit(self, dataloader, epochs=1, fverbose=0, callbacks=None, verbose=1):
        self.fverbose = fverbose
        if self.fverbose > 0:
            if self.fverbose_path is None:
                self.fverbose_path = "training_parameters.txt"
            self.fverbose_file = open(self.fverbose_path, 'w', encoding='utf8')

        self._update_parameters(self.model_branch1.optimizer)
        if self.domain_randomization and self.optimize:
            self._update_parameters(self.model_branch2.optimizer)

        history = []
        try:
            for epoch in range(epochs):
                epoch_logs = [self.train_step(batch) for batch in dataloader]
                averaged = ({k: sum(d[k] for d in epoch_logs) / len(epoch_logs)
                             for k in epoch_logs[0]} if epoch_logs else {})
                history.append(averaged)
                if verbose:
                    shown = " - ".join(f"{k}: {v:.4f}" for k, v in averaged.items())
                    print(f"Epoch {epoch + 1}/{epochs} - {shown}")
                for cb in (callbacks or []):
                    cb(epoch, averaged)
        finally:
            if self.fverbose > 0 and self.fverbose_file is not None:
                self.fverbose_file.close()
                self.fverbose_file = None
        return history

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.eval()
        totals, count = {}, 0
        for imgs, labs in dataloader:
            predictions = self(imgs, training=False)
            batch = {"loss": float(self.loss_fn(predictions, labs))}
            for name, fn in self.metrics_fns.items():
                batch[name] = float(fn(predictions, labs))
            n = labs.shape[0]
            for k, v in batch.items():
                totals[k] = totals.get(k, 0.0) + v * n
            count += n
        return {k: v / count for k, v in totals.items()} if count else {}

    @torch.no_grad()
    def predict(self, dataloader):
        self.eval()
        out = []
        for batch in dataloader:
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            out.append(self(imgs, training=False))
        return torch.cat(out)


    def set_fverbose__file_path(self, filepath):
        self.fverbose_path = filepath

    def get_fverbose__file_path(self):
        return self.fverbose_path

    def _print_fverbose(self, branch=2):
        if self.fverbose_file is None:
            return
        if branch == 1:
            for name, param in self.named_parameters():
                self.fverbose_file.write(f"{name}: {param.detach().cpu().numpy()}\n")
            self.fverbose_file.write("\n\n")
        if branch == 2 and self.parameter_space is not None:
            self.fverbose_file.write(self.parameter_space.describe())
            self.fverbose_file.write("\n\n")



class ResNetBlock(nn.Module):

    def __init__(self, in_channels, filters):
        super().__init__()
        filters1, filters2 = filters
        self.conv2a = nn.Conv2d(in_channels, filters1, 3, padding="same")
        self.conv2b = nn.Conv2d(filters1, filters2, 3, padding="same")

    def forward(self, inputs):
        x = F.relu(self.conv2a(inputs))
        x = F.relu(self.conv2b(x))
        return x + inputs


class ResNet1(ADRModel):

    def __init__(self, n_classes, input_shape=(3, 128, 128), field='data',
                 domain_randomization=None, output_activation=None, verbose=True):
        super().__init__(input_shape=input_shape, field=field,
                         domain_randomization=domain_randomization,
                         output_activation=output_activation, verbose=verbose)
        channels = self.input_shape_[0]
        self.branch1_conv_1 = nn.Conv2d(channels, 32, 3)
        self.branch1_conv_2 = nn.Conv2d(32, 64, 3)
        self.branch1_maxpool = nn.MaxPool2d(3)
        self.branch1_block_1 = ResNetBlock(64, (64, 64))
        self.branch1_block_2 = ResNetBlock(64, (64, 64))
        self.branch1_conv_3 = nn.Conv2d(64, 64, 3)
        self.branch1_glopool = nn.AdaptiveAvgPool2d(1)
        self.branch1_dense_1 = nn.Linear(64, 256)
        self.branch1_do = nn.Dropout(0.5)
        self.branch1_dense_2 = nn.Linear(256, n_classes)

    def _features(self, data):
        x = F.relu(self.branch1_conv_1(data))
        x = F.relu(self.branch1_conv_2(x))
        x = self.branch1_maxpool(x)
        x = self.branch1_block_1(x)
        x = self.branch1_block_2(x)
        x = F.relu(self.branch1_conv_3(x))
        x = torch.flatten(self.branch1_glopool(x), 1)
        x = F.relu(self.branch1_dense_1(x))
        x = self.branch1_do(x)
        return self.branch1_dense_2(x)



def _resnet152_backbone(pretrained=True):
    from torchvision import models
    weights = models.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet152(weights=weights)
    return nn.Sequential(*list(net.children())[:-2])


class _ResNet2Base(ADRModel):

    FREEZE_BACKBONE = False
    DEEP_HEAD = False

    def __init__(self, n_classes, input_shape=(3, 128, 128), field='data',
                 domain_randomization=None, output_activation=None, verbose=True,
                 pretrained=True, base_model=None):
        super().__init__(input_shape=input_shape, field=field,
                         domain_randomization=domain_randomization,
                         output_activation=output_activation, verbose=verbose)

        self.branch1_base_model = (base_model if base_model is not None
                                   else _resnet152_backbone(pretrained))
        if self.FREEZE_BACKBONE:
            for param in self.branch1_base_model.parameters():
                param.requires_grad = False

        self.branch1_flatten = nn.Flatten()
        n_features = self._flatten_features(self.branch1_base_model)
        self.branch1_dense1 = nn.Linear(n_features, 1000)

        if self.DEEP_HEAD:
            self.branch1_dense2 = nn.Linear(1000, 1024)
            self.branch1_dense3 = nn.Linear(1024, 2048)
            self.branch1_dense4 = nn.Linear(2048, 1024)
            self.branch1_dense5 = nn.Linear(1024, n_classes)
        else:
            self.branch1_dense2 = nn.Linear(1000, n_classes)

    def _features(self, data):
        base_model_output = self.branch1_base_model(data)
        x = self.branch1_flatten(base_model_output)
        x = F.relu(self.branch1_dense1(x))
        if not self.DEEP_HEAD:
            return self.branch1_dense2(x)
        x = F.relu(self.branch1_dense2(x))
        x = F.relu(self.branch1_dense3(x))
        x = F.relu(self.branch1_dense4(x))
        return self.branch1_dense5(x)


class ResNet2__0(_ResNet2Base):
    FREEZE_BACKBONE = False
    DEEP_HEAD = False


class ResNet2__1(_ResNet2Base):
    FREEZE_BACKBONE = True
    DEEP_HEAD = False


class ResNet2__0__1(_ResNet2Base):
    FREEZE_BACKBONE = False
    DEEP_HEAD = True


class ResNet2__1__1(_ResNet2Base):
    FREEZE_BACKBONE = True
    DEEP_HEAD = True



class NN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


class NN2(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(NN2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layers)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layers, num_classes)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.relu(x1)
        out = self.fc2(x2)
        return out


class NN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
