
% error('Tracker not configured! Please edit the tracker_Resist.m file.'); % Remove this line after proper configuration

% The human readable label for the tracker, used to identify the tracker in reports
% If not set, it will be set to the same value as the identifier.
% It does not have to be unique, but it is best that it is.
tracker_label = [];

% Now you have to set up the system command to be run.
% For classical executables this is usually just a full path to the executable plus
% optional arguments:

% tracker_command = '<TODO: set a tracker executable command>';
script = '/home/jvlmdr/projects/2017-01-mem-track/rnntracking_dev/track.py';

PYTHON_BIN = '/home/jvlmdr/projects/2017-01-mem-track/rnntracking_dev/env/bin/python';
SRC_DIR = '/home/jvlmdr/projects/2017-01-mem-track/rnntracking_dev';
MODEL_FILE = '/home/jvlmdr/projects/2017-01-mem-track/experiments/2017-11-13-T4-siam-coarse-sc_net-MotionPrior/workspace/ckpt/iteration-160000';

% NOTE: VOT does not support quotes to include spaces.
% (It does not use Matlab's system() call.)
model_flags = {...
  '--model=Nornn' ...
  '--cnn_model=custom' ...
  '--model_params' '{"coarse_hmap":true,"use_hmap_prior":true,"sc":true,"sc_net":true}' ...
};
% name = sprintf('%s-%d-%d', run_opt.test_cfg.seq_name, run_opt.test_cfg.img_start, run_opt.test_cfg.img_end);
vis_flags = {...
  % sprintf('--sequence_name=%s', name) ...
  '--vis' ...
  '--vis_dir=vis' ...
  '--vis_keep_frames' ...
};
extra_flags = {'--gpu_frac=0.45'};

script_path = fullfile(SRC_DIR, 'track.py');
tracker_command = strjoin(...
  [{PYTHON_BIN script_path MODEL_FILE '--vot'} model_flags vis_flags extra_flags], ' ');

% tracker_interpreter = []; % Set the interpreter used here as a lower case string. E.g. if you are using Matlab, write 'matlab'. (optional)

% tracker_linkpath = {}; % A cell array of custom library directories used by the tracker executable (optional)

