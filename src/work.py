import joblib
from box import Box

a = Box(
    {
        "params": {"p": 1},
        "metrics": {"m": 2},
    }
)

joblib.dump(a, "results.pkl")
