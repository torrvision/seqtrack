function results = Run_Resist(imgfilepath_fmt, img_range_str, init_rect, run_opt)

if nargin < 1
  % Platform check.
  results = {};
  return
end

times = eval(img_range_str);
im_size = size(imread(sprintf(imgfilepath_fmt, times(1))));

init_rect_json = convert_rect(init_rect, im_size);

PYTHON_BIN = '/home/jvlmdr/projects/2017-01-mem-track/rnntracking_dev/env/bin/python';
SRC_DIR = '/home/jvlmdr/projects/2017-01-mem-track/rnntracking_dev';
MODEL_FILE = '/home/jvlmdr/projects/2017-01-mem-track/experiments/2017-11-13-T4-siam-coarse-sc_net-MotionPrior/workspace/ckpt/iteration-160000';

flags = ['--model Nornn --cnn_model custom ' ...
  '--model_params ''{"coarse_hmap": true, "use_hmap_prior": true, "sc": true, "sc_net": true}'''];

script_file = fullfile(SRC_DIR, 'track.py');
rects_file = [tempname() '.csv'];
command = sprintf('%s %s %s %s --image_format ''%s'' --start %d --end %d --init_rect ''%s'' %s', ...
  PYTHON_BIN, script_file, MODEL_FILE, rects_file, imgfilepath_fmt, times(1), times(end), init_rect_json, flags);
system(command);

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
