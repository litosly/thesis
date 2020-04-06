from models.ranksvm1 import RankSVM1
from models.ranksvm2 import RankSVM2
from models.ranksvm3 import RankSVM3
from models.lp1_simplified import LP1Simplified
from models.plrec import plrec
from models.average import average


models = {
    "PLRec": plrec,
}

critiquing_models = {
    "ranksvm1": RankSVM1,
    "ranksvm2": RankSVM2,
    "ranksvm3":RankSVM3,
    "average": average,
    "rating": LP1Simplified
}
