import os
import openpyxl
import openml






def write_res(write_path, model_type, acc, odds_difference, cost_time):
    if not os.path.exists(write_path):
        workbook = openpyxl.Workbook()
        res_worksheet = workbook.active

        res_worksheet['D5'] = 'metrics'
        res_worksheet['D6'] = 'Acc'
        res_worksheet['D7'] = 'EO'
        res_worksheet['D8'] = 'Time'
    else:
        workbook = openpyxl.load_workbook(write_path)
        res_worksheet = workbook['Sheet']
    if model_type == 'All Feature':
        res_worksheet['E5'] = model_type
        res_worksheet['E6'] = str(acc)
        res_worksheet['E7'] = str(odds_difference)
        res_worksheet['E8'] = str(cost_time)
    elif model_type == 'OSFS':
        res_worksheet['F5'] = model_type
        res_worksheet['F6'] = str(acc)
        res_worksheet['F7'] = str(odds_difference)
        res_worksheet['F8'] = str(cost_time)
    elif model_type == 'Remove S':
        res_worksheet['G5'] = model_type
        res_worksheet['G6'] = str(acc)
        res_worksheet['G7'] = str(odds_difference)
        res_worksheet['G8'] = str(cost_time)

    elif model_type == 'FS^2-RI':
        res_worksheet['H5'] = model_type
        res_worksheet['H6'] = str(acc)
        res_worksheet['H7'] = str(odds_difference)
        res_worksheet['H8'] = str(cost_time)
    elif model_type == 'FS^2-AD1':
        res_worksheet['I5'] = model_type
        res_worksheet['I6'] = str(acc)
        res_worksheet['I7'] = str(odds_difference)
        res_worksheet['I8'] = str(cost_time)
    elif model_type == 'FS^2-AD2':
        res_worksheet['J5'] = model_type
        res_worksheet['J6'] = str(acc)
        res_worksheet['J7'] = str(odds_difference)
        res_worksheet['J8'] = str(cost_time)
    else:
        raise ValueError("model_type don't have a correct value")

    workbook.save(write_path)
    workbook.close()

