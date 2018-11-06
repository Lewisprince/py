# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 15:41:41 2018

@author: DELL
"""

import pandas as pd
s=pd.Series([1,2,3,4])
print(s.index)
new_index=pd.date_range('2018-1-30',periods=len(s),freq='m')
print(new_index)