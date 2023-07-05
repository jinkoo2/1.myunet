from export_data_parser import parse_export_data
from param import Param, param_from_json, param_from_txt

# string or blank


def _s(dict, key):
    if key in dict.keys():
        return str(dict[key])
    else:
        return ''

# float or 0.0


def _f(dict, key):
    if key in dict.keys():
        return float(dict[key])
    else:
        return 0.0


def update_db_file(data_root, db_json_file):
    print(f'==> update_db_file({data_root})')

    print('Parsing...')
    db = parse_export_data(data_root)

    print('Saving db.json...')
    db.save_to_json(db_json_file)

    print(f'<== update_db_file({data_root})')

def update_plan_list_file(db_json_file):
    db = load_db(db_json_file)
    list = get_plan_list(db)
    param = Param()
    param["list"] = list
    json_file = db_json_file+".plan_list.json"
    print('writing...'+json_file)
    param.save_to_json(json_file)

def update_plan_list_sample_file(db_json_file):
    db = load_db(db_json_file)
    list = get_plan_list(db)
    param = Param()
    param["list"] = list[0:9]
    json_file = db_json_file+".plan_list.sample.json"
    print('writing...'+json_file)
    param.save_to_json(json_file)

def update_contour_list_file(db_json_file):
    db = load_db(db_json_file)
    list = get_contour_list(db)
    param = Param()
    param["list"] = list
    json_file = db_json_file+".contour_list.json"
    print('writing...'+json_file)
    param.save_to_json(json_file)

def update_contour_list_sample_file(db_json_file):
    db = load_db(db_json_file)
    list = get_contour_list(db)
    param = Param()
    param["list"] = list[0:10]
    json_file = db_json_file+".contour_list.sample.json"
    print('writing...'+json_file)
    param.save_to_json(json_file)

def _sum(f_list):
    sum = 0.0
    for f in f_list:
        sum += f
    return sum

def _mean(f_list):

    if len(f_list) == 0:
        return 0.0

    sum = 0.0
    for f in f_list:
        sum += f
    return sum/len(f_list)

# string list


def _s_list(dict_list, key):
    list = []
    for d in dict_list:
        list.append(_s(d, key))
    return list

# string list


def _f_list(dict_list, key):
    list = []
    for d in dict_list:
        list.append(_f(d, key))
    return list


# unique string list
def _u(str_list):
    return list(set(str_list))

def load_db(db_json_file):
    print('loading...:'+db_json_file)
    return param_from_json(db_json_file)

def uuid2pid(db, uuid2pt_file):
    dict_uuid2pt = param_from_txt(uuid2pt_file)
    for pt in db["pt_list"]:
        uuid = pt["Id"]
        if dict_uuid2pt.has_key(uuid):
            pid = dict_uuid2pt[uuid]
            pt["Id"] = pid
        else:
            print(f'Error: missing uuid({uuid}) in {uuid2pt_file}!')          
    return db

def get_plan_list(db):
    
    list = []

    print('parsing db...')

    for pt in db["pt_list"]:
        for ps in pt["ps_list"]:
            beam_list = ps["beam_list"]

            plan = {}

            plan["Patient.Id"] = _s(pt, "Id")
            plan["Course.Id"] = _s(ps, "Course.Id")
            plan["PlanSetup.Id"] = _s(ps, "Id")
            plan["StructureSet.Id"] = _s(ps, "StructureSet.Id")
            plan["Image.Id"] = _s(ps, "Image.Id")
            plan["PlanType"] = _s(ps, "PlanType")
            plan["TreatmentOrientation"] = _s(ps, "TreatmentOrientation")
            plan["PrescribedDosePerFraction"] = _s(
                ps, "PrescribedDosePerFraction").replace('cGy', '').replace('Gy', '').strip()
            plan["NumberOfFractions"] = _s(ps, "NumberOfFractions")
            plan["TotalPrescribedDose"] = _s(ps, "TotalPrescribedDose").replace(
                'cGy', '').replace('Gy', '').strip()
            plan["CreationDateTime"] = _s(ps, "CreationDateTime")
            plan["Beams.Count"] = len(beam_list)
            plan["ExternalBeam.Id"] = '|'.join(
                _u(_s_list(beam_list, "ExternalBeam.Id")))
            plan["MU.Total"] = _sum(_f_list(beam_list, "Meterset.Value"))
            plan["EnergyModeDisplayName"] = '|'.join(
                _u(_s_list(beam_list, "EnergyModeDisplayName")))
            plan["Technique"] = '|'.join(_u(_s_list(beam_list, "Technique")))
            plan["ToleranceTableLabel"] = '|'.join(
                _u(_s_list(beam_list, "ToleranceTableLabel")))
            plan["ControlPoints.Count.PerBeam"] = _mean(
                _f_list(beam_list, "ControlPoints.Count"))
            plan["Blocks.Count.PerBeam"] = _mean(
                _f_list(beam_list, "Blocks.Count"))
            plan["Boluses.Count.PerBeam"] = _mean(
                _f_list(beam_list, "Boluses.Count"))
            plan["DoseRate.PerBeam"] = _mean(_f_list(beam_list, "DoseRate"))
            plan["AverageSSD.PerBeam"] = _mean(
                _f_list(beam_list, "AverageSSD"))
            plan["SSD.PerBeam"] = _mean(_f_list(beam_list, "SSD"))

            list.append(plan)

    return list


# def get_plan_list_as_string_list(db_json_file):
#     print('get_plan_list()')
#     print('loading...:'+db_json_file)
#     db = param_from_json(db_json_file)

#     lines = []

#     # header
#     cols = []
#     cols.append("pt")
#     cols.append("cs")
#     cols.append("plan")
#     cols.append("image")
#     cols.append("PlanType")
#     cols.append("TreatmentOrientation")
#     cols.append("PrescribedDosePerFraction")
#     cols.append("NumberOfFractions")
#     cols.append("TotalPrescribedDose")
#     cols.append("CreationDateTime")
#     cols.append("Beams.Count")
#     cols.append("ExternalBeam.Id")
#     cols.append("MU.Total")
#     cols.append("EnergyModeDisplayName")
#     cols.append("Technique")
#     cols.append("ToleranceTableLabel")
#     cols.append("ControlPoints.Count.PerBeam")
#     cols.append("Blocks.Count.PerBeam")
#     cols.append("Boluses.Count.PerBeam")
#     cols.append("DoseRate.PerBeam")
#     cols.append("AverageSSD.PerBeam")
#     cols.append("SSD.PerBeam")

#     line = ','.join(cols)
#     lines.append(line)

#     print('parsing db...')

#     for pt in db["pt_list"]:
#         for ps in pt["ps_list"]:
#             beam_list = ps["beam_list"]

#             cols = []
#             cols.append(_s(pt, "Id"))
#             cols.append(_s(ps, "Course.Id"))
#             cols.append(_s(ps, "Id").replace(',', '[comma]'))
#             cols.append(_s(ps, "Image.Id"))
#             cols.append(_s(ps, "PlanType"))
#             cols.append(_s(ps, "TreatmentOrientation"))
#             cols.append(_s(ps, "PrescribedDosePerFraction").replace(
#                 'cGy', '').replace('Gy', '').strip())
#             cols.append(_s(ps, "NumberOfFractions"))
#             cols.append(_s(ps, "TotalPrescribedDose").replace(
#                 'cGy', '').replace('Gy', '').strip())
#             cols.append(_s(ps, "CreationDateTime"))
#             cols.append(str(len(beam_list)))
#             cols.append('|'.join(_u(_s_list(beam_list, "ExternalBeam.Id"))))
#             cols.append(str(int(_sum(_f_list(beam_list, "Meterset.Value")))))
#             cols.append(
#                 '|'.join(_u(_s_list(beam_list, "EnergyModeDisplayName"))))
#             cols.append('|'.join(_u(_s_list(beam_list, "Technique"))))
#             cols.append(
#                 '|'.join(_u(_s_list(beam_list, "ToleranceTableLabel"))))
#             cols.append(str(_mean(_f_list(beam_list, "ControlPoints.Count"))))
#             cols.append(str(_mean(_f_list(beam_list, "Blocks.Count"))))
#             cols.append(str(_mean(_f_list(beam_list, "Boluses.Count"))))
#             cols.append(str(_mean(_f_list(beam_list, "DoseRate"))))
#             cols.append(str(_mean(_f_list(beam_list, "AverageSSD"))))
#             cols.append(str(_mean(_f_list(beam_list, "SSD"))))

#             line = ','.join(cols)
#             lines.append(line)

#     return lines


def get_contour_list(db):

    list = []

    print('parsing db...')

    for pt in db["pt_list"]:
        sset_list = pt["sset_list"]
        for ps in pt["ps_list"]:

            # structure set id of the plan
            sset_id = _s(ps, "StructureSet.Id")

            # if plan has not sset, skip.
            if sset_id == '':
                print(
                    f'plan ({ps["Id"]}) has no structure set. pt={pt["Id"]}, cs={ps["Course.Id"]}.')
                continue

            # find sset of the given id.
            sset_found = None
            for sset in sset_list:
                if sset["Id"] == sset_id:
                    sset_found = sset
                    break

            if sset_found == None:
                print(f'sset not found of Id={sset_id}. So skipping!')
                continue

            # sset
            sset = sset_found

            # beams
            beam_list = ps["beam_list"]

            for s in sset["s_list"]:
                try:
                    cont = {}

                    # s info
                    cont["Name"] = _s(s, "name")
                    cont["Volume"] = _f(s, "volume")
                    cont["NumberOfSeparateParts"] = int(s["NumberOfSeparateParts"])
                    cont["Length.X"] = s["MeshGeometry.Bounds"][3]
                    cont["Length.Y"] = s["MeshGeometry.Bounds"][4]
                    cont["Length.Z"] = s["MeshGeometry.Bounds"][5]
                    cont["IsHighResolution"] = s["IsHighResolution"]
                    cont["HistoryUserName"] = s["HistoryUserName"]

                    # pt info
                    cont["Patient.Id"] = _s(pt, "Id")

                    # cs info
                    cont["Course.Id"] = _s(ps, "Course.Id")

                    # ps info
                    cont["PlanSetup.Id"] = _s(ps, "Id")
                    cont["Image.Id"] = _s(ps, "Image.Id")
                    cont["PlanType"] = _s(ps, "PlanType")
                    cont["TreatmentOrientation"] = _s(ps, "TreatmentOrientation")
                    cont["PrescribedDosePerFraction"] = _s(
                        ps, "PrescribedDosePerFraction").replace('cGy', '').replace('Gy', '').strip()
                    cont["NumberOfFractions"] = _s(ps, "NumberOfFractions")
                    cont["TotalPrescribedDose"] = _s(ps, "TotalPrescribedDose").replace(
                        'cGy', '').replace('Gy', '').strip()
                    cont["CreationDateTime"] = _s(ps, "CreationDateTime")
                    cont["Beams.Count"] = len(beam_list)
                    cont["ExternalBeam.Id"] = '|'.join(
                        _u(_s_list(beam_list, "ExternalBeam.Id")))
                    cont["MU.Total"] = _sum(_f_list(beam_list, "Meterset.Value"))
                    cont["EnergyModeDisplayName"] = '|'.join(
                        _u(_s_list(beam_list, "EnergyModeDisplayName")))
                    cont["Technique"] = '|'.join(
                        _u(_s_list(beam_list, "Technique")))
                    cont["ToleranceTableLabel"] = '|'.join(
                        _u(_s_list(beam_list, "ToleranceTableLabel")))
                    cont["ControlPoints.Count.PerBeam"] = _mean(
                        _f_list(beam_list, "ControlPoints.Count"))
                    cont["Blocks.Count.PerBeam"] = _mean(
                        _f_list(beam_list, "Blocks.Count"))
                    cont["Boluses.Count.PerBeam"] = _mean(
                        _f_list(beam_list, "Boluses.Count"))
                    cont["DoseRate.PerBeam"] = _mean(
                        _f_list(beam_list, "DoseRate"))
                    cont["AverageSSD.PerBeam"] = _mean(
                        _f_list(beam_list, "AverageSSD"))
                    cont["SSD.PerBeam"] = _mean(_f_list(beam_list, "SSD"))

                    # sset info
                    cont["StructureSet.Id"] = _s(sset, "Id")


                    list.append(cont)
                except KeyError as e:
                    print(e)
                except Exception as e:
                    print(e)

    return list



