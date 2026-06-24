function paths = project_paths(projectRoot, createMissing)
%PROJECT_PATHS Percorsi canonici e indipendenti dalla current folder.
    if nargin < 2
        createMissing = false;
    end
    projectRoot = char(projectRoot);
    paths = struct('Root', projectRoot, ...
        'Source', fullfile(projectRoot, 'src'), ...
        'Checkpoints', fullfile(projectRoot, 'CHECKPOINTS'), ...
        'Charts', fullfile(projectRoot, 'CHARTS'), ...
        'Logs', fullfile(projectRoot, 'LOGS'), ...
        'Results', fullfile(projectRoot, 'RESULTS'));
    if createMissing
        names = {'Checkpoints', 'Charts', 'Logs', 'Results'};
        for idx = 1:numel(names)
            folder = paths.(names{idx});
            if ~exist(folder, 'dir')
                mkdir(folder);
            end
        end
    end
end
