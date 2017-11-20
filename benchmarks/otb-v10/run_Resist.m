function results=run_Resist(seq, res_path, bSaveImage)

imgfilepath_fmt = fullfile(seq.path, ['%0' num2str(seq.nz) 'd.jpg']);
img_range_str = sprintf('%d:%d', seq.startFrame, seq.endFrame);
init_rect = seq.init_rect;

times = eval(img_range_str);
im_size = size(imread(sprintf(imgfilepath_fmt, times(1))));

init_rect_json = convert_rect(init_rect, im_size);

PYTHON_BIN = '/home/jvlmdr/projects/2017-01-mem-track/rnntracking_dev/env/bin/python';
SRC_DIR = '/home/jvlmdr/projects/2017-01-mem-track/rnntracking_dev';
MODEL_FILE = '/home/jvlmdr/projects/2017-01-mem-track/experiments/2017-11-13-T4-siam-coarse-sc_net-MotionPrior/workspace/ckpt/iteration-160000';

script_file = fullfile(SRC_DIR, 'track.py');
rects_file = 'result_Resist.csv'; % [tempname() '.csv'];
task_flags = {...
  MODEL_FILE ...
  '--out_file' rects_file ...
  quote(sprintf('--image_format=%s', imgfilepath_fmt)) ...
  sprintf('--start=%d', times(1)) ...
  sprintf('--end=%d', times(end)) ...
  quote(sprintf('--init_rect=%s', init_rect_json)) ...
};
model_flags = {...
  '--model=Nornn' ...
  '--cnn_model=custom' ...
  quote('--model_params={"coarse_hmap": true, "use_hmap_prior": true, "sc": true, "sc_net": true}') ...
};
name = sprintf('%s-%d-%d', run_opt.test_cfg.seq_name, run_opt.test_cfg.img_start, run_opt.test_cfg.img_end);
vis_flags = {...
  sprintf('--sequence_name=%s', name) ...
  '--vis' ...
  '--vis_dir=vis' ...
  '--vis_keep_frames' ...
};
extra_flags = {'--gpu_frac=0.45'};

command = strjoin([{PYTHON_BIN, script_file} task_flags model_flags vis_flags extra_flags], ' ');
% fprintf('\n');
% fprintf('%s\n', command);
% fprintf('\n');
status = system(command);
if status ~= 0
  error(['non-zero status: ' num2str(status)]);
end

rects = importdata(rects_file);
h_im = im_size(1);
w_im = im_size(2);
xmin = rects.data(:, 2) * w_im;
ymin = rects.data(:, 3) * h_im;
xmax = rects.data(:, 4) * w_im;
ymax = rects.data(:, 5) * h_im;

x = xmin - 1;
y = ymin - 1;
w = xmax - xmin;
h = ymax - ymin;

results = struct();
results.res = [x, y, w, h];
if size(results.res, 1) == length(times) - 1
  results.res = [init_rect; results.res];
end
results.type = 'rect';
results.fps = 1;

end

function s = convert_rect(rect, im_size)
  x = rect(1) - 1;
  y = rect(2) - 1;
  w = rect(3);
  h = rect(4);
  h_im = im_size(1);
  w_im = im_size(2);

  q = struct();
  q.xmin = x / w_im;
  q.ymin = y / h_im;
  q.xmax = (x+w) / w_im;
  q.ymax = (y+h) / h_im;
  s = jsonencode(q);
end
