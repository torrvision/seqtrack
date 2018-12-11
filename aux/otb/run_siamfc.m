function result = run_siamfc(seq, rp, bSaveImage)

    im_size = size(imread(seq.s_frames{1}));
    init_rect_json = convert_rect(seq.init_rect, im_size);

    images_file = [tempname() '_frames.txt'];
    dump_lines(images_file, seq.s_frames);

    MODEL_FILE = '/home/jvlmdr/projects/2017-01-mem-track/workspace/train/seed_0/ckpt/iteration-80000';
    flags = [...
        '--loglevel=info ' ...
        '--model_params_file /home/jvlmdr/projects/2017-01-mem-track/workspace/model_params.json '...
    ];
    rects_file = [tempname() '_rects.csv'];
    command = sprintf(...
        'python -m seqtrack.tools.track %s --out_file %s --images_file ''%s'' --init_rect ''%s'' %s', ...
        MODEL_FILE, rects_file, images_file, init_rect_json, flags);
    fprintf('command: %s\n', command);
    status = system(command);
    if status ~= 0
        error(sprintf('non-zero return code: %d', status))
    end

    % Skip the first row (header) and column (image file - string).
    rect_data = csvread(rects_file, 1, 1);
    h_im = im_size(1);
    w_im = im_size(2);
    xmin = rect_data(:, 1) * w_im;
    ymin = rect_data(:, 2) * h_im;
    xmax = rect_data(:, 3) * w_im;
    ymax = rect_data(:, 4) * h_im;

    x = xmin - 1;
    y = ymin - 1;
    w = xmax - xmin;
    h = ymax - ymin;

    result = struct();
    result.res = [x, y, w, h];
    if size(result.res, 1) == length(seq.s_frames) - 1
        result.res = [seq.init_rect; result.res];
    end
    result.type = 'rect';
    result.fps = 0;
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
    % s = jsonencode(q);
    s = sprintf(...
      '{"xmin": %g, "ymin": %g, "xmax": %g, "ymax": %g}', ...
      q.xmin, q.ymin, q.xmax, q.ymax);
end


function dump_lines(filename, lines)
    f = fopen(filename, 'w');
    fprintf(f, '%s\n', lines{:});
    fclose(f);
end
