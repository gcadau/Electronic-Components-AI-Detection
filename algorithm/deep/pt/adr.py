import numpy as np
import torch

from ..utils.common import (
    denormalize_value,
    iterate_over_elements_below_diagonal,
    normalize_value,
    set_parameters__initials,
    set_parameters__ranges,
    spiral_flat_from_progressive,
)
from ..utils.pt import Triangular, fill_triangular, is_nevergrad_optimizer, scaleAndFlat_matrix

try:
    import nevergrad as ng
except ImportError:  
    ng = None




_MODE_GROUPS = {
    "uniform": (("lowers", "lowers"), ("uppers", "uppers")),
    "triangular": (("lowers", "lowers"), ("modes", "modes"), ("uppers", "uppers")),
    "univariate normal": (("means", "means"), ("variances", "variances")),
    "multivariate normal": (("mean vector", "mean_vector"),
                            ("variancecovariance_matrix", "variancecovariance_matrix")),
}

_RANGE_MODE = {
    "uniform": "uniform",
    "triangular": "triangular",
    "univariate normal": "univariatenormal",
    "multivariate normal": "multivariatenormal",
}

_INF_SUBSTITUTE = 1000

_LETTERS = "ABCDEFG"

_TF_SUFFIX = {
    "lowers": "lower", "uppers": "upper", "modes": "mode",
    "means": "mean", "variances": "variance",
}


class Branch:

    def __init__(self, name="default", module=None):
        self.name = name
        self.module = module
        self.optimizer = None
        self._tr_vars = []

    def get_name(self):
        return self.name

    def add_variables(self, var):
        if isinstance(var, (list, tuple)):
            self._tr_vars.extend(var)
        else:
            self._tr_vars.append(var)

    def get_trainable_variables(self):
        return self._tr_vars


class ADRParameterSpace:

    def __init__(self, domain_randomization, n_parameters=7, normalized_space=None,
                 tf_legacy_triangular=False):
        self.mode = domain_randomization.mode
        if self.mode not in _MODE_GROUPS:
            raise ValueError(f"unknown domain randomization mode: {self.mode}")

        self.n_parameters = n_parameters
        self.normalized_space = normalized_space or {"lower": 0.0, "upper": 4.0}
        self.seed = domain_randomization.seed
        self.tf_legacy_triangular = tf_legacy_triangular

        self.groups = tuple(g for g, _ in _MODE_GROUPS[self.mode])
        self.ranges, self.initials = {}, {}
        for group, attr in _MODE_GROUPS[self.mode]:
            rng = getattr(domain_randomization, f"{attr}__ranges", None)
            if rng is None:
                rng = set_parameters__ranges(_RANGE_MODE[self.mode], group)
            self.ranges[group] = rng
            init = getattr(domain_randomization, f"{attr}__initials", None)
            if init is None:
                init = set_parameters__initials(_RANGE_MODE[self.mode], group, rng)
            self.initials[group] = self._sanitize_initials(init, rng)

        self.values = {g: list(self.initials[g]) for g in self.groups}

        if self.mode == "multivariate normal":
            self._spiral = spiral_flat_from_progressive(
                len(self.initials["variancecovariance_matrix"]))
            self._chol_flat = list(
                scaleAndFlat_matrix(np.array(self.initials["variancecovariance_matrix"])))
            self.values["variancecovariance_matrix"] = list(self._chol_flat)

        self._scalars = None
        self._pending = None


    @staticmethod
    def _sanitize_initials(initials, ranges):
        out = []
        for value, (low, high) in zip(initials, ranges):
            if np.isfinite(value):
                out.append(value)
                continue
            low = -_INF_SUBSTITUTE if low == float('-inf') else low
            high = _INF_SUBSTITUTE if high == float('inf') else high
            out.append((high + low) / 2)
        return out

    def _range_for(self, group, k):
        if group == "variancecovariance_matrix":
            return self.ranges[group][self._spiral[k]]
        return self.ranges[group][k]

    def _flat_initials(self):
        for group in self.groups:
            source = (self._chol_flat if group == "variancecovariance_matrix"
                      else self.initials[group])
            for k, value in enumerate(source):
                yield group, k, value

    def parametrization(self):
        if ng is None:  
            raise ImportError("nevergrad is required to optimize the domain randomization.")
        scalars = []
        for group, k, value in self._flat_initials():
            low, high = self._range_for(group, k)
            init = normalize_value(value, low, high,
                                   self.normalized_space["lower"],
                                   self.normalized_space["upper"])
            init = min(max(init, self.normalized_space["lower"]),
                       self.normalized_space["upper"])
            scalars.append(
                ng.p.Scalar(init=init).set_bounds(
                    lower=self.normalized_space["lower"],
                    upper=self.normalized_space["upper"])
            )
        self._scalars = scalars
        return ng.p.Instrumentation(params=ng.p.Tuple(*scalars))

    def ask(self, optimizer):
        candidate = optimizer.ask()
        flat = list(candidate.kwargs["params"])
        i = 0
        for group in self.groups:
            n = len(self.values[group])
            out = []
            for k in range(n):
                low, high = self._range_for(group, k)
                out.append(denormalize_value(flat[i + k], low, high,
                                             self.normalized_space["lower"],
                                             self.normalized_space["upper"]))
            self.values[group] = out
            i += n
        self._pending = candidate
        return candidate

    def tell(self, optimizer, loss):
        if self._pending is not None:
            optimizer.tell(self._pending, float(loss))


    def sample(self, n, generator=None):
        if self.mode == "multivariate normal":
            loc = torch.as_tensor(self.values["mean vector"], dtype=torch.float32)
            scale_tril = fill_triangular(
                torch.as_tensor(self.values["variancecovariance_matrix"], dtype=torch.float32))
            eps = torch.randn((n, loc.shape[0]), generator=generator)
            return loc + eps @ scale_tril.T

        out = []
        for k in range(self.n_parameters):
            if self.mode == "uniform":
                low = self.values["lowers"][k]
                high = self.values["uppers"][k]
                out.append(torch.rand(n, generator=generator) * (high - low) + low)
            elif self.mode == "triangular":
                out.append(Triangular(self.values["lowers"][k],
                                      self.values["modes"][k],
                                      self.values["uppers"][k],
                                      tf_legacy=self.tf_legacy_triangular,
                                      generator=generator).sample((n,)))
            else:  
                mean = self.values["means"][k]
                scale = self.values["variances"][k]
                out.append(torch.randn(n, generator=generator) * scale + mean)
        return tuple(out)


    @staticmethod
    def _rescaling_factor(ranges_a, ranges_b):
        width = torch.tensor([b[1] - a[0] for a, b in zip(ranges_a, ranges_b)],
                             dtype=torch.float32)
        width = torch.where(torch.isinf(width), torch.full_like(width, 1000.0), width)
        return width / width.max()

    def penalty(self, epsilon=0.001):
        zero = torch.tensor(0.0)
        if self.mode == "uniform":
            rf = self._rescaling_factor(self.ranges["lowers"], self.ranges["uppers"])
            terms = [torch.maximum(zero, torch.tensor(
                self.values["lowers"][i] - self.values["uppers"][i] + epsilon)) / rf[i]
                for i in range(self.n_parameters)]
            return torch.stack(terms).mean()

        if self.mode == "triangular":
            total = torch.tensor(0.0)
            rf = self._rescaling_factor(self.ranges["lowers"], self.ranges["modes"])
            total = total + torch.stack([torch.maximum(zero, torch.tensor(
                self.values["lowers"][i] - self.values["modes"][i] + epsilon)) / rf[i]
                for i in range(self.n_parameters)]).mean()
            rf = self._rescaling_factor(self.ranges["modes"], self.ranges["uppers"])
            total = total + torch.stack([torch.maximum(zero, torch.tensor(
                self.values["modes"][i] - self.values["uppers"][i] + epsilon)) / rf[i]
                for i in range(self.n_parameters)]).mean()
            return total

        if self.mode == "multivariate normal":
            sigma = self.covariance()
            ranges = self.ranges["variancecovariance_matrix"]
            width = torch.tensor([r[1] - r[0] for r in ranges], dtype=torch.float32)
            width = torch.where(torch.isinf(width), torch.full_like(width, 1000.0), width)
            rf = width / width.max()
            below = iterate_over_elements_below_diagonal(sigma.shape[0])
            over = torch.stack([torch.maximum(zero, sigma[i, j] - ranges[index][1] + epsilon) / rf[index]
                                for index, (i, j) in enumerate(below)]).mean()
            under = torch.stack([torch.maximum(zero, ranges[index][0] - sigma[i, j] + epsilon) / rf[index]
                                 for index, (i, j) in enumerate(below)]).mean()
            return over + under

        return torch.tensor(0.0)  

    def covariance(self):
        """Rebuild SIGMA = L @ L.T from the optimized Cholesky factor."""
        chol = fill_triangular(
            torch.as_tensor(self.values["variancecovariance_matrix"], dtype=torch.float32))
        return chol @ chol.T


    def __getattr__(self, item):
        if item.startswith("branch2_"):
            rest = item[len("branch2_"):]
            if rest in ("mean_vector", "variancecovariance_matrix"):
                key = "mean vector" if rest == "mean_vector" else rest
                return object.__getattribute__(self, "values")[key]
            base, letter = rest[:-1], rest[-1]
            if letter in _LETTERS:
                for group, suffix in _TF_SUFFIX.items():
                    if suffix == base and group in object.__getattribute__(self, "values"):
                        return object.__getattribute__(self, "values")[group][_LETTERS.index(letter)]
        raise AttributeError(item)

    def variable_names(self):
        if self.mode == "multivariate normal":
            return ["branch2_mean_vector", "branch2_variancecovariance_matrix"]
        names = []
        for group in self.groups:
            suffix = _TF_SUFFIX[group]
            names += [f"branch2_{suffix}{_LETTERS[i]}" for i in range(self.n_parameters)]
        return names

    def describe(self):
        lines = []
        if self.mode == "uniform":
            for i in range(self.n_parameters):
                lines.append(f"Param {i}: U({self.values['lowers'][i]}, {self.values['uppers'][i]})")
        elif self.mode == "triangular":
            for i in range(self.n_parameters):
                lines.append(f"Param {i}: Tr({self.values['lowers'][i]}, "
                             f"{self.values['modes'][i]}, {self.values['uppers'][i]})")
        elif self.mode == "univariate normal":
            for i in range(self.n_parameters):
                lines.append(f"Param {i}: N({self.values['means'][i]}, {self.values['variances'][i]})")
        else:
            mu = np.array(self.values["mean vector"])
            sigma = self.covariance().numpy()
            lines.append("Params [1,...,i,...,n]\u1d40: N(mu, SIGMA)")
            lines.append(f"\tmu = {mu}")
            rows = [" ".join(map(str, row)) for row in sigma]
            lines.append("\tSIGMA = [" + rows[0])
            for row in rows[1:-1]:
                lines.append("\t         " + row)
            lines.append("\t         " + rows[-1] + "]")
        return "\n".join(lines) + "\n"


def build_dr_layers(domain_randomization, parameters_name, optimize):
    from ...domain_randomization.pt import (
        r_multivariatenormal, r_triangular, r_uniform, r_univariatenormal)
    from ...domain_randomization.optimization.pt import (
        r_multivariatenormal as r_multivariatenormal_opt,
        r_triangular as r_triangular_opt,
        r_uniform as r_uniform_opt,
        r_univariatenormal as r_univariatenormal_opt)

    mode = domain_randomization.mode
    modules = {
        "uniform": (r_uniform, r_uniform_opt),
        "triangular": (r_triangular, r_triangular_opt),
        "univariate normal": (r_univariatenormal, r_univariatenormal_opt),
        "multivariate normal": (r_multivariatenormal, r_multivariatenormal_opt),
    }[mode]
    module = modules[1] if optimize else modules[0]

    if mode == "multivariate normal":
        params = {"factors": domain_randomization.factors, "seed": domain_randomization.seed}
        if not optimize:
            params["mean_vector"] = domain_randomization.mean_vector
            params["variancecovariance_matrix"] = domain_randomization.variancecovariance_matrix
        return {"parameters": module.RandomParameters(**params)}

    per_group = {"uniform": ("lowers", "uppers"),
                 "triangular": ("lowers", "modes", "uppers"),
                 "univariate normal": ("means", "variances")}[mode]
    kw_name = {"lowers": "lower", "uppers": "upper", "modes": "mode",
               "means": "mean", "variances": "variance"}

    params = {option: {} for option in parameters_name}
    for option in parameters_name:
        idx = parameters_name.index(option)
        if not optimize:
            for group in per_group:
                values = getattr(domain_randomization, group, None)
                if values is not None:
                    params[option][kw_name[group]] = values[idx]
        if domain_randomization.factors is not None:
            params[option]["factor"] = domain_randomization.factors[idx]
        params[option]["seed"] = domain_randomization.seed

    return {
        "brightness": module.RandomBrightness(**params["brightness"]),
        "contrast": module.RandomContrast(**params["contrast"]),
        "horizontally flip": module.RandomHorizontallyFlip(**params["horizontally flip"]),
        "vertically flip": module.RandomVerticallyFlip(**params["vertically flip"]),
        "hue": module.RandomHue(**params["hue"]),
        "jpeg quality": module.RandomJpegQuality(**params["jpeg quality"]),
        "saturation": module.RandomSaturation(**params["saturation"]),
    }
