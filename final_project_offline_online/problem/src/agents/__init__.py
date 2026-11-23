from .fql_agent import FQLAgent
from .sacbc_agent import SACBCAgent
from .qsm_agent import QSMAgent
from .dsrl_agent import DSRLAgent
from .ifql_agent import IFQLAgent
from .qcfql_agent import QCFQLAgent
from .acifql_agent import ACIFQLAgent
from .qam_agent import QAMAgent

agents = {
    "fql": FQLAgent,
    "sacbc": SACBCAgent,
    "qsm": QSMAgent,
    "dsrl": DSRLAgent,
    "ifql": IFQLAgent,
    "qcfql": QCFQLAgent,
    "acifql": ACIFQLAgent,
    "qam": QAMAgent,
}
 