function results = run_hpa_benchmark(varargin)
%RUN_HPA_BENCHMARK BER vs SNR, MMSE contro CNN, per piu livelli IBO.

    parser = inputParser();
    addParameter(parser, 'ProjectRoot', "");
    addParameter(parser, 'IBODb', []);
    addParameter(parser, 'SNRDb', []);
    addParameter(parser, 'Smoothness', 2);
    addParameter(parser, 'NumPackets', []);
    addParameter(parser, 'MinFrames', 20);
    addParameter(parser, 'MaxFrames', 200);
    addParameter(parser, 'TargetBitErrors', 500);
    addParameter(parser, 'IncludeCNN', true);
    addParameter(parser, 'ModelPath', "");
    addParameter(parser, 'MakePlots', true);
    addParameter(parser, 'WriteLog', true);
    addParameter(parser, 'SaveResults', true);
    addParameter(parser, 'FigureVisible', 'on');
    addParameter(parser, 'Profile', 'standard');
    addParameter(parser, 'Seed', 352);
    parse(parser, varargin{:});
    cfg = parser.Results;

    projectRoot = resolve_project_root(cfg.ProjectRoot);
    paths = project_paths(projectRoot, true);
    params = simulation_parameters().refreshDerived();
    params.SNRDefinition = 'SNR';
    profile = validatestring(char(cfg.Profile), {'standard', 'smoke'});
    if isempty(cfg.IBODb)
        cfg.IBODb = params.TestIBODb;
    end
    if isempty(cfg.SNRDb)
        if strcmp(profile, 'smoke')
            cfg.SNRDb = [0 18 34];
        else
            cfg.SNRDb = params.TestSNRDb;
        end
    end
    if strcmp(profile, 'smoke')
        if ~isempty(cfg.NumPackets)
            cfg.NumPackets = min(cfg.NumPackets, 1);
        end
        cfg.MinFrames = min(cfg.MinFrames, 1);
        cfg.MaxFrames = min(cfg.MaxFrames, 1);
        cfg.TargetBitErrors = min(cfg.TargetBitErrors, 1);
    end
    validate_config(cfg);

    carrierMap = params.getCarrierMap();
    frameTool = IEEE80211aFrame(params);
    methodNames = "MMSE";
    cnn = [];
    modelPath = "";
    if cfg.IncludeCNN
        [cnn, modelPath] = load_hpa_cnn(params, cfg.ModelPath, paths.Checkpoints);
        methodNames(end+1) = "CNN";
    end

    iboDb = double(cfg.IBODb(:)).';
    snrDb = double(cfg.SNRDb(:)).';
    numIBO = numel(iboDb);
    numSNR = numel(snrDb);
    numMethods = numel(methodNames);
    errors = zeros(numIBO, numSNR, numMethods);
    totalBits = zeros(numIBO, numSNR);
    numFrames = zeros(numIBO, numSNR);
    saturationFraction = zeros(numIBO, numSNR);
    actualIBODb = zeros(numIBO, numSNR);

    print_header(cfg, iboDb, snrDb, modelPath, methodNames);
    for iboIdx = 1:numIBO
        fprintf("\nIBO %+.1f dB\n", iboDb(iboIdx));
        fprintf(" SNR      Sat.    Frame       Bit");
        for methodIdx = 1:numMethods
            fprintf("   BER %-8s", methodNames(methodIdx));
        end
        fprintf("\n");

        for snrIdx = 1:numSNR
            params.SNR_dB = snrDb(snrIdx);
            pointErrors = zeros(1, numMethods);
            framesSeen = 0;
            bitsSeen = 0;
            saturationSum = 0;
            actualIBOSum = 0;

            if isempty(cfg.NumPackets)
                frameLimit = cfg.MaxFrames;
            else
                frameLimit = cfg.NumPackets;
            end

            while framesSeen < frameLimit
                % Frame e rumore normalizzato identici lungo lo sweep SNR.
                rng(double(cfg.Seed) + 100000 * iboIdx + framesSeen, 'twister');
                [txWaveform, txFrame] = frameTool.createRandomFrame();
                [rxWaveform, hpaInfo] = apply_hpa_impairments( ...
                    txWaveform, params, iboDb(iboIdx), cfg.Smoothness);
                rxFrame = frameTool.receive(rxWaveform);
                framesSeen = framesSeen + 1;

                [mmseGrid, ~] = equalize_mmse_pilots(rxFrame.ActiveGrid, ...
                    txFrame.PilotSymbols, carrierMap, rxFrame.NoiseVariance);
                mmseSymbols = mmseGrid(carrierMap.PayloadActiveIdx, :);
                pointErrors(1) = pointErrors(1) + count_bit_errors( ...
                    mmseSymbols, txFrame.PayloadBits, params.ModulationOrder);

                if cfg.IncludeCNN
                    X = build_frame_cnn_batch(rxFrame, txFrame, ...
                        carrierMap, cnn.NumInputChannels);
                    X = cnn.prepareInputBatch(X, carrierMap.ActiveGlobalIdx);
                    scores = gather(extractdata(forward(cnn.Net, ...
                        dlarray(single(X), 'CTB'))));
                    scores = normalize_score_shape(scores, cnn, ...
                        params.NumOFDMSymbols);
                    predictedClasses = cnn.oneHotToClasses(scores);
                    predictedClasses = predictedClasses( ...
                        carrierMap.PayloadGlobalIdx, :);
                    predictedBits = cnn.classesToBits(predictedClasses);
                    pointErrors(2) = pointErrors(2) + ...
                        sum(predictedBits(:) ~= txFrame.PayloadBits(:));
                end

                saturationSum = saturationSum + hpaInfo.SaturatedSampleFraction;
                actualIBOSum = actualIBOSum + hpaInfo.ActualInputBackOffDb;
                bitsSeen = bitsSeen + numel(txFrame.PayloadBits);
                if isempty(cfg.NumPackets) && framesSeen >= cfg.MinFrames && ...
                        all(pointErrors >= cfg.TargetBitErrors)
                    break;
                end
            end

            errors(iboIdx, snrIdx, :) = pointErrors;
            totalBits(iboIdx, snrIdx) = bitsSeen;
            numFrames(iboIdx, snrIdx) = framesSeen;
            saturationFraction(iboIdx, snrIdx) = saturationSum / framesSeen;
            actualIBODb(iboIdx, snrIdx) = actualIBOSum / framesSeen;
            fprintf(" %5.1f dB  %5.1f%%  %6d  %9d", snrDb(snrIdx), ...
                100 * saturationFraction(iboIdx, snrIdx), framesSeen, bitsSeen);
            fprintf("   %10.3e", pointErrors ./ bitsSeen);
            fprintf("\n");
        end
    end

    timestamp = string(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    ber = errors ./ max(totalBits, 1);
    plotBER = max(errors, 0.5) ./ max(totalBits, 1);
    results = struct();
    results.Timestamp = timestamp;
    results.ChannelModel = "RAPP_AMAM_PLUS_FLAT_AWGN";
    results.DataContract = model_data_contract();
    results.IBODb = iboDb;
    results.SNRDb = snrDb;
    results.Smoothness = double(cfg.Smoothness);
    results.MethodNames = methodNames;
    results.Errors = errors;
    results.TotalBits = totalBits;
    results.NumFrames = numFrames;
    results.BER = ber;
    results.PlotBER = plotBER;
    results.SaturatedSampleFraction = saturationFraction;
    results.ActualIBODb = actualIBODb;
    results.ModelPath = modelPath;
    results.Parameters = params;
    results.ZeroErrorConvention = "0.5/numero_bit, solo visualizzazione";
    results.TestConfig = cfg;
    if cfg.IncludeCNN
        mmseIdx = find(methodNames == "MMSE", 1);
        cnnIdx = find(methodNames == "CNN", 1);
        results.CNNBetterFraction = mean( ...
            results.BER(:, :, cnnIdx) < results.BER(:, :, mmseIdx), 'all');
        results.CNNNotWorseFraction = mean( ...
            results.BER(:, :, cnnIdx) <= results.BER(:, :, mmseIdx), 'all');
    else
        results.CNNBetterFraction = NaN;
        results.CNNNotWorseFraction = NaN;
    end

    chartInfo = struct('Files', strings(0, 1));
    if cfg.MakePlots
        [~, chartInfo] = plot_hpa_results(results, paths.Charts, ...
            cfg.FigureVisible);
    end
    logPath = "";
    if cfg.WriteLog
        logPath = write_test_log(results, chartInfo, paths.Logs);
    end
    resultPath = string(fullfile(paths.Results, ...
        sprintf('rs_%s.mat', char(timestamp))));
    if ~cfg.SaveResults
        resultPath = "";
    end
    results.OutputFiles = struct('Charts', chartInfo.Files, ...
        'Log', logPath, 'Results', resultPath);

    if cfg.IncludeCNN
        fprintf("\nCNN migliore della MMSE nel %.1f%% dei punti.\n", ...
            100 * results.CNNBetterFraction);
        fprintf("CNN migliore o uguale nel %.1f%% dei punti.\n", ...
            100 * results.CNNNotWorseFraction);
    end
    if cfg.SaveResults
        save(resultPath, 'results', '-v7.3');
    end
    fprintf("Completato: %d IBO x %d livelli SNR.\n", numIBO, numSNR);
end

function validate_config(cfg)
    iboDb = double(cfg.IBODb(:));
    snrDb = double(cfg.SNRDb(:));
    if isempty(iboDb) || any(~isfinite(iboDb))
        error('IBODb deve contenere valori finiti.');
    end
    if isempty(snrDb) || any(~isfinite(snrDb)) || any(diff(snrDb) <= 0)
        error('SNRDb deve essere finito e strettamente crescente.');
    end
    if ~isscalar(cfg.Smoothness) || cfg.Smoothness <= 0
        error('Smoothness deve essere positivo.');
    end
    if ~isempty(cfg.NumPackets) && (~isscalar(cfg.NumPackets) || ...
            ~isfinite(cfg.NumPackets) || cfg.NumPackets < 1 || ...
            cfg.NumPackets ~= round(cfg.NumPackets))
        error('NumPackets deve essere vuoto oppure un intero positivo.');
    end
    if cfg.MinFrames < 1 || cfg.MaxFrames < cfg.MinFrames || ...
            any([cfg.MinFrames cfg.MaxFrames] ~= round([cfg.MinFrames cfg.MaxFrames]))
        error('Richiesto 1 <= MinFrames <= MaxFrames, entrambi interi.');
    end
    if cfg.TargetBitErrors < 1 || cfg.TargetBitErrors ~= round(cfg.TargetBitErrors)
        error('TargetBitErrors deve essere un intero positivo.');
    end
end

function print_header(cfg, iboDb, snrDb, modelPath, methodNames)
    fprintf("=== TEST BER VS SNR | HPA RAPP + AWGN ===\n");
    fprintf("Rapp p=%.2g | IBO: %s dB\n", cfg.Smoothness, ...
        strjoin(compose('%+.0f', iboDb), ', '));
    fprintf("SNR: %s dB | metodi: %s\n", ...
        strjoin(compose('%.0f', snrDb), ', '), strjoin(methodNames, ' / '));
    if isempty(cfg.NumPackets)
        fprintf("Pacchetti: adattivi, %d..%d per punto | target errori: %d\n", ...
            cfg.MinFrames, cfg.MaxFrames, cfg.TargetBitErrors);
    else
        fprintf("Pacchetti: %d per ogni coppia IBO/SNR\n", cfg.NumPackets);
    end
    if strlength(modelPath) > 0
        fprintf("CNN: %s\n", modelPath);
    end
end

function count = count_bit_errors(symbols, trueBits, modOrder)
    bits = qamdemod(symbols(:), modOrder, 'gray', ...
        'UnitAveragePower', true, 'OutputType', 'bit');
    count = sum(uint8(bits(:)) ~= trueBits(:));
end

function [cnn, modelPath] = load_hpa_cnn(params, requestedPath, checkpointDir)
    modelPath = string(requestedPath);
    if strlength(modelPath) == 0
        files = dir(fullfile(checkpointDir, 'ck_*.mat'));
        if isempty(files)
            error('Nessun checkpoint in %s. Eseguire run_trainer.', checkpointDir);
        end
        [~, order] = sort(string({files.name}), 'descend');
        modelPath = string(fullfile(checkpointDir, files(order(1)).name));
    end
    loaded = load(modelPath, 'net', 'trainingState');
    if ~isfield(loaded, 'net') || ~isfield(loaded, 'trainingState')
        error('Checkpoint non valido: %s.', modelPath);
    end
    state = loaded.trainingState;
    if string(state.networkArchitecture) ~= HPA_CNN.architectureVersion() || ...
            ~isfield(state, 'dataContract') || ...
            string(state.dataContract) ~= model_data_contract() || ...
            string(state.distortionModel) ~= "RAPP_AMAM"
        error('Checkpoint incompatibile con pipeline Rapp senza normalizzazione.');
    end
    cnn = HPA_CNN('NFFT', params.FFTLength, ...
        'ModOrder', params.ModulationOrder, ...
        'NumInputChannels', params.NumCNNInputChannels);
    cnn.Net = loaded.net;
end

function scores = normalize_score_shape(scores, cnn, batchCount)
    if ismatrix(scores)
        scores = reshape(scores, size(scores, 1), size(scores, 2), 1);
    end
    if size(scores, 1) == cnn.ModOrder && ...
            size(scores, 2) == cnn.NFFT && size(scores, 3) == batchCount
        return;
    end
    if size(scores, 1) == cnn.ModOrder && ...
            size(scores, 2) == batchCount && size(scores, 3) == cnn.NFFT
        scores = permute(scores, [1 3 2]);
        return;
    end
    error('Shape output CNN non valida: [%s].', num2str(size(scores)));
end

function root = resolve_project_root(requestedRoot)
    root = string(requestedRoot);
    if strlength(root) == 0
        root = string(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    end
    root = char(root);
end

function value = model_data_contract()
    value = "rapp_abs_drive_no_power_norm_v1";
end
