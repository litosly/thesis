from models.ranksvm1 import RankSVM1
from models.ranksvm2 import RankSVM2
from models.plrec import plrec
from models.average import average


models = {
    "PLRec": plrec,
}

critiquing_models = {
    "ranksvm1": RankSVM1,
    "ranksvm2": RankSVM2,
    "average": average
}
