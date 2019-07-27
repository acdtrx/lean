import lean_utils as lu
import csv
from collections import Counter
from matplotlib import pyplot as plt

rdc = Counter()
current_day = 0
next_day = 86400
with open( './cache/auth_red.txt' , 'r' ) as csv_file:
    reader = csv.reader( csv_file )
    for line in reader:
        if int( line[0] ) > next_day:
            current_day += 1
            next_day = (current_day + 1) * 86400
        rdc[current_day] += 1

print( rdc )
x , y = zip( *sorted( rdc.items() ) )
plt.bar( x , y  )
plt.xticks( x , x , rotation="vertical" )
plt.show()
