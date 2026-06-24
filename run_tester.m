function results = run_tester(varargin)
%RUN_TESTER Entry point del benchmark BER vs SNR.

    %% CONFIGURAZIONE UTENTE - modificare direttamente questi valori
    IBO_DB = [-10 -5 0 5];   % una coppia MMSE/CNN per ogni valore
    SNR_DB = 0:2:36;            % punti dell'asse SNR
    NUM_PACKETS = 300;           % pacchetti valutati per ogni coppia IBO/SNR

    projectRoot = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(projectRoot, 'src')));

    defaults = {'ProjectRoot', projectRoot, ...
        'IBODb', IBO_DB, ...
        'SNRDb', SNR_DB, ...
        'NumPackets', NUM_PACKETS};
    arguments = merge_name_value(defaults, varargin);
    results = run_hpa_benchmark(arguments{:});
end

function output = merge_name_value(defaults, overrides)
% Permette sia l'editing del blocco sopra sia override da command window.
    if mod(numel(overrides), 2) ~= 0
        error('Gli override devono essere coppie nome/valore.');
    end
    output = defaults;
    for idx = 1:2:numel(overrides)
        name = string(overrides{idx});
        defaultNames = string(output(1:2:end));
        match = find(strcmpi(defaultNames, name), 1);
        if isempty(match)
            output(end+1:end+2) = overrides(idx:idx+1);
        else
            output{2 * match} = overrides{idx+1};
        end
    end
end
