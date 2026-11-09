"""Immutable per-run controls; legacy reports use the original strict defaults."""
from dataclasses import asdict, dataclass
import argparse
import math


@dataclass(frozen=True)
class FitPolicy:
    support_optimization: bool = True
    penetration_optimization: bool = True
    support_validation: bool = False
    penetration_validation: bool = False
    support_min_coverage: float = .90
    support_gluteus_gap_m: float = .015
    support_thigh_gap_m: float = .020
    max_penetration_m: float = .005

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name.endswith(("optimization", "validation")):
                if type(value) is not bool:
                    raise ValueError(name + " must be boolean")
            elif not math.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if not 0 < self.support_min_coverage <= 1:
            raise ValueError("support_min_coverage must be in (0, 1]")

    @property
    def strict(self):
        return self.support_validation and self.penetration_validation

    def gap(self, region):
        return self.support_gluteus_gap_m if "gluteus" in region else self.support_thigh_gap_m

    def record(self):
        return dict(version=1, **asdict(self))

    @classmethod
    def from_record(cls, value=None):
        # Missing fields in historical reports mean the original strict policy.
        # New runs use cls()/from_args() and always serialize all fields.
        values = dict(support_validation=True, penetration_validation=True)
        values.update({k:v for k,v in (value or {}).items() if k != "version"})
        return cls(**values)

    @classmethod
    def from_args(cls, args):
        values = {}
        for name, default in asdict(cls()).items():
            key = name[:-2] + "_mm" if name.endswith("_m") else name
            value = getattr(args, key, default * 1000 if name.endswith("_m") else default)
            values[name] = value / 1000 if name.endswith("_m") else value
        return cls(**values)

    def argv(self):
        result=[]
        for name,value in asdict(self).items():
            key=name.replace("_","-")
            if type(value) is bool:
                result.append("--" + ("" if value else "no-") + key)
            else:
                if name.endswith("_m"):
                    key=key[:-2]+"-mm"; value*=1000
                result += ["--"+key,str(value)]
        return result


def add_arguments(parser):
    for name in ("support-optimization", "penetration-optimization", "support-validation", "penetration-validation"):
        parser.add_argument("--"+name, action=argparse.BooleanOptionalAction,
                            default=getattr(FitPolicy(), name.replace("-", "_")))
    for name,value in (("support-min-coverage",.90),("support-gluteus-gap-mm",15.),
                       ("support-thigh-gap-mm",20.),("max-penetration-mm",5.)):
        parser.add_argument("--"+name,type=float,default=value)


def eligible(candidate):
    return candidate.get("selection_eligible", candidate.get("qualified", False))


def skipped():
    return dict(status="skipped_disabled", checked_source_frames=0)
