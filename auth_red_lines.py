import csv

from tqdm import tqdm

from lean_params import gen_params
import lean_utils as lu

redteam_filename = './data/redteam.txt'
input_filename = './data/auth_users.txt'

auth_red_filename = f'./cache/auth_red.txt'

rt_file = open( redteam_filename , 'r' )
rt_csv = csv.reader( rt_file )

ar_file = open( auth_red_filename , 'w' )
ar_csv = csv.writer( ar_file )
with open( input_filename , 'r' ) as i_file:
    rt_line = next( rt_csv )
    (rt_time , rt_src_user , rt_src_comp , rt_dst_comp ) = rt_line
    rt_time = int( rt_time )
    p_bar = tqdm( csv.reader( i_file ) , desc="RT find" , total=gen_params['ws_size'] )
    rt_lines_found = 0
    rt_lines_lost = 0
    for auth_line in p_bar:
        (i_time , i_src_usr , i_dst_user , i_src_comp , i_dst_comp , i_auth_type , i_logon_type , i_auth_ori , i_success ) = auth_line
        i_time = int( i_time )
        # p_bar.set_postfix( rt_lines = rt_lines_found , i_time = i_time , refresh=False)
        if i_time == rt_time and i_src_usr == rt_src_user and i_src_comp == rt_src_comp and i_dst_comp == rt_dst_comp:
            #found one redteam sample
            p_bar.set_postfix( rt_lines = rt_lines_found , i_time = i_time , refresh=False)
            ar_csv.writerow( auth_line )
            rt_lines_found += 1
            rt_line = next( rt_csv , None )
            if rt_line == None:
                print( 'rt_done found' )
                break
            (rt_time , rt_src_user , rt_src_comp , rt_dst_comp ) = rt_line
            rt_time = int( rt_time )
        if i_time > rt_time:
            rt_lines_lost += 1
            rt_line = next( rt_csv , None )
            if rt_line == None:
                print( 'rt_done lost' )
                break
            (rt_time , rt_src_user , rt_src_comp , rt_dst_comp ) = rt_line
            rt_time = int( rt_time )

print( f'Found {rt_lines_found} Lost {rt_lines_lost}' )
ar_file.close()
rt_file.close()
