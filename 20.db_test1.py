from export_data_parser import parse_export_data
from db_service import update_plan_list_file, update_contour_list_file, update_plan_list_sample_file, update_contour_list_sample_file

data_root = 'c:\\data'
db_json_file = 'c:\\data\\db.json'
sample_pt_json_file = 'c:\\data\\_sample_pt.json'

print(f'==> update_db_file({data_root})')

# print('Parsing...')
# db = parse_export_data(data_root)

# ############################
# # make db.json file
# print('Saving db.json...')
# db.save_to_json(db_json_file)

# # save a sample pt to a file for review
# print('Saving a sample pt file...:', sample_pt_json_file)
# db["pt_list"][0].save_to_json(sample_pt_json_file)

###################################
# update paln list json file
print('Updating plan list json file')
# update_plan_list_file(db_json_file)
update_plan_list_sample_file(db_json_file)

############################
# update contour list file
print('Updating contour list json file')
# update_contour_list_file(db_json_file)
update_contour_list_sample_file(db_json_file)

print('done')
