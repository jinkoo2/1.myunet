from db_service import load_db, get_contour_list, get_plan_list, uuid2pid
from dict_helper import dict_list_to_string_list
from os_helper import write_all_text

##################
# db
db_json = 'c:\\data\\db.json'
uuid2pid_file = 'c:\\data\\uuid2pt.txt'
db = load_db(db_json)

# replace uuid with pid
print('replacing uuids with pids')
db = uuid2pid(db, uuid2pid_file)

####################
# plan list to file
csv = "c:\\data\\plans.csv"
list = get_plan_list(db)
lines = dict_list_to_string_list(list)
print(f'N={len(lines)}')
print(f'saving to {csv}...')
write_all_text(csv, '\n'.join(lines))

######################
# contour list to file
csv = "c:\\data\\contours.csv"
list = get_contour_list(db)

lines = dict_list_to_string_list(list)
print(f'N={len(lines)}')
print(f'saving to {csv}...')
write_all_text(csv,'\n'.join(lines))

# print(len(lines))
print('done')
