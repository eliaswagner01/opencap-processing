'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_kinetics.py
    ---------------------------------------------------------------------------
    Copyright 2022 Stanford University and the Authors
    
    Author(s): Antoine Falisse, Scott Uhlrich
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    
    This code makes use of CasADi, which is licensed under LGPL, Version 3.0;
    https://github.com/casadi/casadi/blob/master/LICENSE.txt.
    
    Install requirements:
        - Visit https://github.com/stanfordnmbl/opencap-processing for details.        
        - Third-party software packages:
            - CMake: https://cmake.org/download/.
            - (Windows only)
                - Visual studio: https://visualstudio.microsoft.com/downloads/.
                    - Make sure you install C++ support.
                    - Code tested with community editions 2017-2019-2022.
            
    Please contact us for any questions: https://www.opencap.ai/#contact
'''

# %% Directories, paths, and imports. You should not need to change anything.
import os
import sys

baseDir = os.path.dirname(os.path.abspath(__file__))
opensimADDir = os.path.join(baseDir, 'UtilsDynamicSimulations', 'OpenSimAD')
if baseDir not in sys.path:
    sys.path.append(baseDir)
if opensimADDir not in sys.path:
    sys.path.append(opensimADDir)

from UtilsDynamicSimulations.OpenSimAD.utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD
from UtilsDynamicSimulations.OpenSimAD.mainOpenSimAD import run_tracking
from utils import download_kinematics

# %% User inputs.
'''
Please provide:
    
    session_id:     This is a 36 character-long string. You can find the ID of
                    all your sessions at https://app.opencap.ai/sessions.
                    
    trial_name:     This is the name of the trial you want to simulate. You can
                    find all trial names after loading a session.
                    
    motion_type:    This is the type of activity you want to simulate. Options
                    are 'running', 'walking', 'drop_jump', 'sit-to-stand', and
                    'squats'. We provide pre-defined settings that worked well
                    for this set of activities. If your activity is different,
                    select 'other' to use generic settings or set your own
                    settings in settingsOpenSimAD. See for example how we tuned
                    the 'running' settings to include periodic constraints in
                    the 'my_periodic_running' settings.
                    
    time_window:    This is the time interval you want to simulate. It is
                    recommended to simulate trials shorter than 2s. Set to []
                    to simulate full trial. For 'squats' or 'sit_to_stand', we
                    built segmenters to separate the different repetitions. In
                    such case, instead of providing the time_window, you can
                    provide the index of the repetition (see below) and the
                    time_window will be automatically computed.
                    
    repetition:     Only if motion_type is 'sit_to_stand' or 'squats'. This
                    is the index of the repetition you want to simulate (0 is 
                    first). There is no need to set the time_window. 
                    
    case:           This is a string that will be appended to the file names
                    of the results. Dynamic simulations are optimization
                    problems, and it is common to have to play with some
                    settings to get the problem to converge or converge to a
                    meaningful solution. It is useful to keep track of which
                    solution corresponds to which settings; you can then easily
                    compare results generated with different settings.
                    
    (optional)
    treadmill_speed:This an optional parameter that indicates the speed of
                    the treadmill in m/s. A positive value indicates that the
                    subject is moving forward. You should ignore this parameter
                    or set it to 0 if the trial was not measured on a
                    treadmill. By default, treadmill_speed is set to 0.
    (optional)
    contact_side:   This an optional parameter that indicates on which foot to
                    add contact spheres to model foot-ground contact. It might
                    be useful to only add contact spheres on one foot if only
                    that foot is in contact with the ground. We found this to
                    be helpful for simulating for instance single leg dropjump
                    as it might prevent the optimizer to cheat by using the
                    other foot to stabilize the model. Options are 'all', 
                    'left', and 'right'. By default, contact_side is set to
                    'all', meaning that contact spheres are added to both feet.
    
See example inputs below for different activities. Please note that we did not
verify the biomechanical validity of the results; we only made sure the
simulations converged to kinematic solutions that were visually reasonable.

Please contact us for any questions: https://www.opencap.ai/#contact
'''

session_id = "3375ffbc-daeb-4a43-b4f7-ac9899cd4c71"
case = 'nonperiodic' # Change this to compare across settings.

# Specify trial names in a list; use None to process all trials in a session.
# These are the trials currently listed in:
# C:\Users\wagnerel85475\Documents\Thesis\opencap-core\Data\3375ffbc-daeb-4a43-b4f7-ac9899cd4c71\OpenSimData\Kinematics
specific_trial_names = [
    'sit-to-stand1',
    'sit-to-stand2',
    'sit-to-stand3',
]

# Default settings used for any trial that is not listed in trial_settings.
# Set motion_type to the best match for your session before running.
default_trial_settings = {
    'motion_type': 'drop_jump',
    'time_window': [],
    'repetition': None,
    'treadmill_speed': 0,
    'contact_side': 'all',
}

# Per-trial settings. Leave time_window as [] and repetition as None until you
# define the window or repetition you want for each trial.
trial_settings = {
    'drop-jump2': {
        'motion_type': 'drop_jump',
        'time_window': [2.45, 3.7],
        'repetition': None,
        'treadmill_speed': 0,
        'contact_side': 'all'
    },
    'drop-jump3': {
        'motion_type': 'drop_jump',
        'time_window': [2.3, 3.8],
        'repetition': None,
        'treadmill_speed': 0,
        'contact_side': 'all'
    },
    'sit-to-stand1': {
        'motion_type': 'sit_to_stand',
        'time_window': [1.0, 2.4],
        'repetition': None,
        'treadmill_speed': 0,
        'contact_side': 'all'
    },
    'sit-to-stand2': {
        'motion_type': 'sit_to_stand',
        'time_window': [1.5, 2.9],
        'repetition': None,
        'treadmill_speed': 0,
        'contact_side': 'all'
    },
    'sit-to-stand3': {
        'motion_type': 'sit_to_stand',
        'time_window': [1.2, 2.7],
        'repetition': None,
        'treadmill_speed': 0,
        'contact_side': 'all'
    },
    'squat1': {
        'motion_type': 'squats',
        'time_window': [1.5, 4.0],
        'repetition': None,
        'treadmill_speed': 0,
        'contact_side': 'all'
    },
    'squat2': {
        'motion_type': 'squats',
        'time_window': [6.4, 8.5],
        'repetition': None,
        'treadmill_speed': 0,
        'contact_side': 'all'
    },
    'walk1': {
        'motion_type': 'walking',
        'time_window': [3.8, 5.2],
        'repetition': None,
        'treadmill_speed': 0,
        'contact_side': 'all'
    },
    'walk2': {
        'motion_type': 'walking',
        'time_window': [9.25, 10.7],
        'repetition': None,
        'treadmill_speed': 0,
        'contact_side': 'all'
    },
    'walk3': {
        'motion_type': 'walking',
        'time_window': [9.95, 11.5],
        'repetition': None,
        'treadmill_speed': 0,
        'contact_side': 'all'
    },
}

# Set to True to solve the optimal control problem.
solveProblem = True
# Set to True to analyze the results of the optimal control problem. If you
# solved the problem already, and only want to analyze/process the results, you
# can set solveProblem to False and run this script with analyzeResults set to
# True. This is useful if you do additional post-processing but do not want to
# re-run the problem.
analyzeResults = True

# Set to True to plot results.
plotResults = False

# Set to True to only generate the OpenSimAD model/contact/expression graph
# functions needed later by run_tracking. This skips the optimization and also
# skips repetition auto-segmentation for squats/sit-to-stand trials.
generateFunctionsOnly = False

# Path to where you want the data to be downloaded.
dataFolder = os.path.join(baseDir, 'Data')

# %% Setup. 
sessionFolder = os.path.join(dataFolder, session_id)
trial_names, _ = download_kinematics(session_id, folder=sessionFolder,
                                     trialNames=specific_trial_names)

for trial_name in trial_names:
    settings_for_trial = default_trial_settings.copy()
    settings_for_trial.update(trial_settings.get(trial_name, {}))

    motion_type = settings_for_trial['motion_type']
    time_window = settings_for_trial['time_window']
    repetition = settings_for_trial['repetition']
    treadmill_speed = settings_for_trial['treadmill_speed']
    contact_side = settings_for_trial['contact_side']

    print('Processing trial {} with motion_type={}, time_window={}, '
          'repetition={}'.format(
              trial_name, motion_type, time_window, repetition))

    setup_repetition = None if generateFunctionsOnly else repetition
    settings = processInputsOpenSimAD(baseDir, dataFolder, session_id,
                                      trial_name, motion_type, time_window,
                                      setup_repetition, treadmill_speed,
                                      contact_side)

    if motion_type == 'sit_to_stand':
        settings.pop('periodicConstraints', None)

    if generateFunctionsOnly:
        model_folder = os.path.join(dataFolder, session_id, 'OpenSimData',
                                    'Model')
        trial_model_folder = os.path.join(model_folder, trial_name)
        if os.path.exists(trial_model_folder):
            model_folder = trial_model_folder
        external_function_folder = os.path.join(model_folder,
                                                'ExternalFunction')
        print('Generated functions are saved in: {}'.format(
            external_function_folder))
        continue

    # %% Simulation.
    run_tracking(baseDir, dataFolder, session_id, settings, case=case,
                 solveProblem=solveProblem,
                 analyzeResults=analyzeResults)

    # %% Plots.
    # To compare different cases, add to the cases list, eg cases=['0','1'].
    if plotResults:
        plotResultsOpenSimAD(dataFolder, session_id, trial_name, settings,
                             cases=[case])
