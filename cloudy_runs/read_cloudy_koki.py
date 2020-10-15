# 10/13/2020
# copied from Koki (https://github.com/enigma-igm/xcorr_QSOfield/blob/master/cloudy/plot_cloudy.py)
# slight modifications, including to print statement

import pandas as pd
def read_cloudy(filename):
    """
        read CLOUDY outputs (*.avr and *.ovr) and make a lookup table
        of various chemical elements as function of density and temperature
        usage: e.g. lookup=read_cloudy(filename='hm12_z6_lookup_table') # note without the ".in" extension
                    output: lookup [pandas table]
    """

    print('reading ... : ', filename)

    avr=pd.read_csv(filename+'.avr',sep='\t') # load average ionization fraction of various speices
    avr=avr.drop(avr.columns[0],axis=1)       # drop the first column as it is nan
    ovr=pd.read_csv(filename+'.ovr',sep='\t') # load overview file, which inlcude hydrogen density (hden [1/cm3]) and temperature (Te [K])

    lookup=pd.concat([ovr,avr],axis=1,sort=False) # combine .ovr and .avr outputs into a summary CLOUDY lookup table

    print('----- Available column names ----- ')
    print(lookup.columns.values)

    return lookup

