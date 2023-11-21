"""Goal of this script:

Processing and plotting the data under the figures in the following publications.

Yu Han, David Salido-Monzú, Andreas Wieser, "Classification of material and surface roughness using polarimetric multispectral LiDAR," Opt. Eng. 62(11) 114104 (21 November 2023)
https://doi.org/10.1117/1.OE.62.11.114104

Date: 21.11.2023
Author: Yu Han
"""



""" ********************************************* Importation ********************************************* """
# ---------------------------------------- Import packages ----------------------------------------
import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import os

# ---------------------------------------- Import functions ----------------------------------------

def plotting_individual_features_subfolder(subfolder):
    # Define font
    font = {'family': 'Arial',
            'size': '13',
            'weight': 'normal'}
    matplotlib.rc('font', **font)

    # create new subfolders
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    # Plot all R, DoLP, AoLP (individual figures)
    R = [R_standard,  DoLP,  R_unpol, R_pol]
    R_title = ['R', 'DoLP',  'R_unpol', 'R_pol']
    y_label = ['$R$',  'DoLP', 'R$_{unpol}$', 'R$_{pol}$']

    # Create different colors and markders for different materials
    color_1 = cm.get_cmap('Paired').colors
    color = color_1
    marker_1 = ["none", "s", "o", "v", "^"]
    marker = []
    for cc_index, (cc,mar) in enumerate(zip(color_1,marker_1)):
        for rr in roughness:
            # color.append(cc)
            marker.append(mar)
    
    # Plot        
    for m, RR in enumerate(R):
        fig, axs = plt.subplots(figsize=(6, 3))
        plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9, wspace=0, hspace=0)
        for n, (sam, sam_name) in enumerate(zip(samples, samples_name)):
            # Calculate mean and std values
            indices = [id for id, ele in enumerate(labels_samples) if ele == n]
            mean = np.mean(RR[indices, :], axis=0)
            std = np.std(RR[indices, :], axis=0)
            # Plotting
            if n % 2 == 0:
                axs.plot(WL, mean, '-', color=color[n], marker=marker[n], markersize=3, label=sam_name)
            else:
                axs.plot(WL, mean, ':', color=color[n], marker=marker[n], markersize=3, label=sam_name)
            axs.fill_between(WL, mean + std, mean - std, facecolor=color[n], alpha=0.2)
            axs.set_xlabel('$\lambda$ [nm]',fontsize=13)
            axs.set_xlim([550, 950])
            axs.set_xticks(np.linspace(550, 950, num=9).astype(int))
            axs.set_xticklabels(np.linspace(550, 950, num=9).astype(int),fontsize=13)
            axs.set_ylabel(str(y_label[m]),fontsize=13)
        axs.set_ylim([0, 1])
        axs.set_yticks(np.linspace(0, 1, num=3))
        axs.set_yticklabels(np.linspace(0, 1, num=3))
        axs.grid(True, alpha=0.4)
        if SCs_select == 1:
            axs.set_title('Spectral configuration [BW = 40 nm]',fontsize=13)
        elif SCs_select == 2:
            axs.set_title('Spectral configuration [BW = 10 nm]',fontsize=13)

        # Save figures with legend
        fig.set_size_inches(12, 5)
        plt.subplots_adjust(left=0.3, bottom=0.3, right=0.7, top=0.95, wspace=0, hspace=0)
        # pass handle & labels lists along with order as below
        axs.legend( bbox_to_anchor=(-0.3, -0.2),
                   loc='upper left',
                   borderaxespad=0.2, ncol=5,fontsize=13)

        if SCs_select == 1:
            filename = os.path.join(subfolder, 'Fig_' + R_title[m] + '[BW=40nm].png')
        elif SCs_select == 2:
            filename = os.path.join(subfolder, 'Fig_' + R_title[m] + '[BW=10nm].png')
        fig.savefig(filename, dpi=600)

    print('Feature plotting done!')


def select_spectral_configuration(SCs_select):
    if SCs_select == 3:
        WL_index_min = 0
        WL_index_max = 40
        SCs_name = 'all 40 SCs'
    elif SCs_select == 1:
        WL_index_min = 0
        WL_index_max = 7
        SCs_name = 'spectral configuration 1'
    elif SCs_select == 2:
        WL_index_min = 7
        WL_index_max = 40
        SCs_name = 'spectral configuration 2'
    else:
        print("Oops!  That was not a valid number.  Please input 1 or 2 to select spectral configuration 1 or 2. ")
    WL_array = np.concatenate((np.linspace(600, 900, num=7), np.linspace(580, 900, num=33)), axis=0)
    WL = WL_array[WL_index_min:WL_index_max]
    return WL, WL_index_min, WL_index_max, SCs_name


def import_data():

    R_standard = np.zeros((len(samples) * Nr_MeasPerSample, WL_index_max - WL_index_min))
    DoLP = np.zeros((len(samples) * Nr_MeasPerSample, WL_index_max - WL_index_min))
    R_unpol = np.zeros((len(samples) * Nr_MeasPerSample, WL_index_max - WL_index_min))
    R_pol = np.zeros((len(samples) * Nr_MeasPerSample, WL_index_max - WL_index_min))


    labels_samples = np.zeros(len(samples) * Nr_MeasPerSample)
    labels_roughness = np.zeros(len(samples) * Nr_MeasPerSample)
    labels_materials = np.zeros(len(samples) * Nr_MeasPerSample)

    # Importing data
    for sam, sample in enumerate(samples):

        # Standard reflectance (from standard measurement without rotation linear polarizer P2)
        fileID = r'Data\Data_' + sample + '_R_normal_SR.txt'
        A = np.transpose(np.loadtxt(fileID, delimiter=','))
        R_standard[sam * Nr_MeasPerSample:(sam + 1) * Nr_MeasPerSample, :] = A[:, WL_index_min:WL_index_max]

        # Degree of linear polarization (DoLP)
        fileID = r'Data\Data_' + sample + '_DoLP.txt'
        A = np.transpose(np.loadtxt(fileID, delimiter=','))
        DoLP[sam * Nr_MeasPerSample:(sam + 1) * Nr_MeasPerSample, :] = A[:, WL_index_min:WL_index_max]

        # unpolarized reflectance (calculated by AoLP)
        fileID = r'Data\Data_' + sample + '_R_unpol.txt'
        A = np.transpose(np.loadtxt(fileID, delimiter=','))
        R_unpol[sam * Nr_MeasPerSample:(sam + 1) * Nr_MeasPerSample, :] = A[:, WL_index_min:WL_index_max]

        # polarized reflectance (calculated by AoLP)
        fileID = r'Data\Data_' + sample + '_R_pol.txt'
        A = np.transpose(np.loadtxt(fileID, delimiter=','))
        R_pol[sam * Nr_MeasPerSample:(sam + 1) * Nr_MeasPerSample, :] = A[:, WL_index_min:WL_index_max]

        # 10 samples are with labels 0-9
        labels_samples[sam * Nr_MeasPerSample:(sam + 1) * Nr_MeasPerSample] = sam * np.ones((Nr_MeasPerSample))
        # roughness levels: P80 with label 0; P400 with label 1
        labels_roughness[sam * Nr_MeasPerSample:(sam + 1) * Nr_MeasPerSample] = (sam % 2) * np.ones((Nr_MeasPerSample))
        # materials: PE(red) with label 0; PVC(red) with label 1; PVC(black) with label 2; PP(pink) with label 3
        labels_materials[sam * Nr_MeasPerSample:(sam + 1) * Nr_MeasPerSample] = (sam // 2) * np.ones((Nr_MeasPerSample))
    print('Data imported!')
    return R_unpol, R_pol, R_standard, DoLP, labels_samples, labels_roughness, labels_materials

def classification_CV_multiple_features_DIY(R, labels_target, labels_assistant, label_type, C_set, tol_set, max_iter_set):
    score = [int for int in range(0, len(R))]
    score_mean = [int for int in range(0, len(R))]
    score_std = [int for int in range(0, len(R))]

    lsvc = LinearSVC(dual=False, C=C_set, penalty = 'l2', tol=tol_set, max_iter=max_iter_set)

    [label_assistant_value, label_assistant_count] = np.unique(labels_assistant, return_counts=True)

    R_title = ['$R$', 'DoLP','R$_{unpol}$', 'R$_{pol}$']

    # Cross-validation DIY
    for m, RR in enumerate(R):
        scores_list = []
        print('***'+label_type+': '+ R_title[m])
        for label_assistant_index, label_assistant in enumerate(label_assistant_value):
            test_index = np.where(labels_assistant == label_assistant)[0]
            full_index = np.arange(0, np.shape(labels_assistant)[0]).astype(int)
            train_index = np.delete(full_index, np.s_[test_index], 0)
            X_train, X_test, y_train, y_test = RR[train_index], RR[test_index], labels_target[train_index], labels_target[test_index]
            lsvc.fit(X_train, y_train)
            y_predict = lsvc.predict(X_test)
            scores_list.append(balanced_accuracy_score(y_test, y_predict))
        score[m] = np.array(scores_list)
        score_mean[m] = np.mean(score[m])
        score_std[m] = np.std(score[m])
    score_mean = np.array(score_mean)
    score_std = np.array(score_std)
    return score_mean, score_std


def plotting_accuracies_subfolder(subfolder, score_mean, score_std, label_type):
    # Define font
    font = {'family': 'Arial',
            'size': '13',
            'weight': 'normal'}
    matplotlib.rc('font', **font)
    # create new subfolders
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    R_title = ['$R$', 'DoLP', 'R$_{unpol}$','R$_{pol}$']

    fig, axs = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.85, wspace=0, hspace=0)
    axs.plot()
    axs.plot(np.arange(0, len(R)), score_mean, '-', color='blue')
    axs.fill_between(np.arange(0, len(R)), score_mean + score_std, score_mean - score_std, facecolor='blue', alpha=0.2)
    axs.set_ylabel('Accuracy score')
    axs.set_xticks(range(0, len(R)))
    axs.set_xticklabels(R_title)
    axs.set_xlabel('Reflectance at different directions')
    if SCs_select == 1:
        axs.set_title(label_type + ' [BW = 40 nm]', fontsize=13)
    elif SCs_select == 2:
        axs.set_title(label_type + ' [BW = 10 nm]', fontsize=13)
    axs.set_ylim([0, 1.3])
    axs.grid(True, alpha=0.4)
    for i, txt in enumerate(score_mean):
        axs.annotate('ave=' + str(np.round(txt, 2)), (i - 0.05, np.sum(score_mean[i] + 0.2)))

    for i, txt in enumerate(score_std):
        axs.annotate('std=' + str(np.round(txt, 2)), (i - 0.05, np.sum(score_mean[i] - 0.2)))

    print(label_type + ' plotting done!')

    if SCs_select == 1:
        filename = os.path.join(subfolder, 'Fig_' + label_type + '[BW=40nm].png')
    elif SCs_select == 2:
        filename = os.path.join(subfolder, 'Fig_' + label_type + '[BW=10nm].png')
    fig.savefig(filename, dpi=600)


""" ++++++++++++++++++++++++++++++++++++++++++++++++++ Spectral configuration 1 (7 spectral channels with 40 nm bandwidth) ++++++++++++++++++++++++++++++++++++++++++++++++++  """
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Spectral configuration 1 (7 spectral channels with 40 nm bandwidth) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

# ---------------------------------------- Import data ----------------------------------------
Nr_MeasPerSample = 20
# Material samples (file name)
samples = ['PP_pink_P80', 'PP_pink_P400',
           'PE_red_P80', 'PE_red_P400',
           'PVC_red_P80', 'PVC_red_P400',
           'sandstone_green_P80', 'sandstone_green_P400',
           'limestone_P80', 'limestone_P400']
# Material samples (sample name)
samples_name = ['PP P80', 'PP P400',
                'PE P80', 'PE P400',
                'PVC P80', 'PVC P400',
                'sandstone P80', 'sandstone P400',
                'limestone P80', 'limestone P400']
# Materials
materials = ['PP_pink', 'PE_red', 'PVC_red', 'sandstone(green)', 'limestone']
# Roughness
roughness = ['P80', 'P400']

# Spectral configuration
SCs_select = 1
WL, WL_index_min, WL_index_max, _ = select_spectral_configuration(SCs_select)

# Importing data
R_unpol, R_pol, R_standard, DoLP, labels_samples, labels_roughness, labels_materials = import_data()
R = [R_standard, DoLP, R_unpol, R_pol]

# Plot data
subfolder = 'Results_Features&Classification'
plotting_individual_features_subfolder(subfolder)


""" ********************************************* Classification ********************************************* """
# ------------------------------------ Material classification ------------------------------------
print('---------------------------------Material classification ---------------------------------')

C_set = 0.1
tol_set = 1e-3
max_iter_set = 1e5

score_mean, score_std = classification_CV_multiple_features_DIY(R, labels_materials,labels_roughness, 'Material classification', C_set,
                                                           tol_set, max_iter_set)
plotting_accuracies_subfolder(subfolder,score_mean, score_std, 'Material classification')

print('Material classification ---> accuracy mean: ', score_mean)
print('Material classification ---> accuracy std: ', score_std)

# ------------------------------------ Roughness classification ------------------------------------
print('---------------------------------Roughness classification ---------------------------------')

C_set = 0.1
tol_set = 1e-3
max_iter_set = 1e5

score_mean, score_std = classification_CV_multiple_features_DIY(R, labels_roughness, labels_materials, 'Roughness classification',
                                                            C_set, tol_set, max_iter_set)
plotting_accuracies_subfolder(subfolder, score_mean, score_std, 'Roughness classification')

print('Roughness classification ---> accuracy mean: ', score_mean)
print('Roughness classification ---> accuracy std: ', score_std)



""" ++++++++++++++++++++++++++++++++++++++++++++++++++ Spectral configuration 2 (33 spectral channels with 10 nm bandwidth) ++++++++++++++++++++++++++++++++++++++++++++++++++  """
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Spectral configuration 2 (33 spectral channels with 10 nm bandwidth) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')


# ---------------------------------------- Import data ----------------------------------------
Nr_MeasPerSample = 20
# Material samples (file name)
samples = ['PP_pink_P80', 'PP_pink_P400',
           'PE_red_P80', 'PE_red_P400',
           'PVC_red_P80', 'PVC_red_P400',
           'sandstone_green_P80', 'sandstone_green_P400',
           'limestone_P80', 'limestone_P400']
# Material samples (sample name)
samples_name = ['PP P80', 'PP P400',
                'PE P80', 'PE P400',
                'PVC P80', 'PVC P400',
                'sandstone P80', 'sandstone P400',
                'limestone P80', 'limestone P400']
# Materials
materials = ['PP_pink', 'PE_red', 'PVC_red', 'sandstone(green)', 'limestone']
# Roughness
roughness = ['P80', 'P400']

# Spectral configuration
SCs_select = 2
WL, WL_index_min, WL_index_max, _ = select_spectral_configuration(SCs_select)

# Importing data
R_unpol, R_pol, R_standard, DoLP, labels_samples, labels_roughness, labels_materials = import_data()
R = [R_standard, DoLP, R_unpol, R_pol]

# Plot data
subfolder = 'Results_Features&Classification'
plotting_individual_features_subfolder(subfolder)

""" ********************************************* Classification ********************************************* """
# ------------------------------------ Material classification ------------------------------------
print('---------------------------------Material classification ---------------------------------')

C_set = 0.1
tol_set = 1e-3
max_iter_set = 1e5

score_mean, score_std = classification_CV_multiple_features_DIY(R, labels_materials,labels_roughness, 'Material classification', C_set,
                                                           tol_set, max_iter_set)
plotting_accuracies_subfolder(subfolder,score_mean, score_std, 'Material classification')

print('Material classification ---> accuracy mean: ', score_mean)
print('Material classification ---> accuracy std: ', score_std)

# ------------------------------------ Roughness classification ------------------------------------
print('---------------------------------Roughness classification ---------------------------------')

C_set = 0.1
tol_set = 1e-3
max_iter_set = 1e5

score_mean, score_std = classification_CV_multiple_features_DIY(R, labels_roughness, labels_materials, 'Roughness classification',
                                                            C_set, tol_set, max_iter_set)
plotting_accuracies_subfolder(subfolder, score_mean, score_std, 'Roughness classification')

print('Roughness classification ---> accuracy mean: ', score_mean)
print('Roughness classification ---> accuracy std: ', score_std)







plt.show()