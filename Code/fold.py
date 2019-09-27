from sklearn.model_selection import PredefinedSplit

def predefined_fold():
    File_Precent = ["10", "30", "50", "70", "90"]
    File_Purpose = ["10fold", "nontest", "test"]
    File_PathPerfix = "/Volumes/Macintosh HD 2/FSU/2018Fall/SeminarJose/Seminar2/FinalProject/Supplementary_Materials/"
    # First, we create Train_Index and Test_Index. Predefined_Split should be established
    # on the list Train_Index.
    Train_Index = []
    Test_Index = []
    for Indi, i in enumerate(File_Precent):
      # - Temp_Dat: Temporary stroage for predefine fold file. Type changeable.
        # First Train_Index
        Temp_Dat = open( File_PathPerfix + i + "percentTest-" + File_Purpose[1], 'r').read().split('\n')[0].split()[1:]
        for Indj, j in enumerate(Temp_Dat):
            Temp_Dat[Indj] = int(j[4:])
        Train_Index.append(Temp_Dat)
        # Then Test_Index
        Temp_Dat = open( File_PathPerfix + i + "percentTest-" + File_Purpose[2], 'r').read().split('\n')[0].split()[1:]
        for Indj, j in enumerate(Temp_Dat):
            Temp_Dat[Indj] = int(j[4:])
        Test_Index.append(Temp_Dat)
        
    # Then, we need to create object sklearn.model_selection.PredefinedSplit
    Predefined_Split = []
    # Loop for test percent
    for Indi, i in enumerate(File_Precent):
      # - Temp_Dat: Temporary stroage for predefine fold file. Type changeable.
      # - Temp_TestFold: Definition of test_fold for PredefinedSplit.
        # 1. Read files and initialization
        Temp_Dat = open( File_PathPerfix + i + "percentTest-" + File_Purpose[0], 'r').read().split('\n')[1:-1]
        # 2. Create Temp_Testfold array
        Temp_TestFold = [0] * len(Train_Index[Indi])
        # Loop for the number of fold
        for Indj, j in enumerate(Temp_Dat):
          # - Temp_Row: Temporary stroage for bare fold file.
          # - Temp_Num_Train: Current training set count.
          # - Temp_Num_Val: Current validation set count.
            Temp_Row = j.split()
            Temp_Num_Train = int(Temp_Row[0])
            Temp_Num_Val = int(Temp_Row[1])
            Temp_Row = Temp_Row[2:]
            # Fill values into the Temp_Testfold array
            # Loop for the validation set in the current fold
            for Indk in range(Temp_Num_Train, Temp_Num_Train + Temp_Num_Val):
              # - Temp_Ind: Index of the current QM9 Id to the Train_Index array.
                Temp_Ind = Train_Index[Indi].index(int(Temp_Row[Indk][4:]))
                Temp_TestFold[Temp_Ind] = Indj
        # 3. Create the split object
        Predefined_Split.append(PredefinedSplit(Temp_TestFold))
        # debug
        #print("--- Show if Temp_TestFold correct ---")
        #for i in range(10):
        #    print(" %s: %s" % (i, Temp_TestFold.count(i)))
    return (Predefined_Split, Train_Index, Test_Index)