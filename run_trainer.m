function summary = run_trainer(varargin)
%RUN_TRAINER Entry point del training HPA_CNN.
    projectRoot = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(projectRoot, 'src')));
    summary = train_hpa_cnn('ProjectRoot', projectRoot, varargin{:});
end
