#

def check_feature_feasible(frame):
    # Check if there is null values
    if frame.isnull().any().any():
        namelist = frame.columns[frame.isnull().any()]
        print "There are null values in the DataFrame:",
        print ", ".join(namelist)
    else:
        print "There is no null values in the DataFrame"
    # Check is there is object type
    if ('object' in frame.dtypes):
        namelist = frame.columns[frame.dtype == 'object']
        print "There are object types in the DataFrame:",
        print ", ".join(namelist)
    else:
        print "There is no object type in the DataFrame"