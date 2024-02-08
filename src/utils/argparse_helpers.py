from pathlib import Path
from typing import List, Dict, Union, Optional

def float_or_int(num:str):
    try:
        return int(num)
    except ValueError:
        return float(num)
    
