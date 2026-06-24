function summary = train_hpa_cnn(varargin)
%TRAIN_HPA_CNN Addestra la CNN invariata su Rapp con curriculum IBO adattivo.

    parser = inputParser();
    addParameter(parser, 'ProjectRoot', "");
    addParameter(parser, 'ResumeModelPath', "");
    addParameter(parser, 'StartPhase', []);
    addParameter(parser, 'CurriculumDirection', "auto");
    addParameter(parser, 'Profile', "standard");
    addParameter(parser, 'Seed', 17889);
    parse(parser, varargin{:});
    cfg = parser.Results;

    projectRoot = resolve_project_root(cfg.ProjectRoot);
    paths = project_paths(projectRoot, true);
    profile = validatestring(char(cfg.Profile), {'standard', 'smoke'});
    requestedDirection = validatestring(char(cfg.CurriculumDirection), ...
        {'auto', 'saturation-to-linear', 'linear-to-saturation'});
    rng(double(cfg.Seed), 'twister');

    params = simulation_parameters().refreshDerived();
    carrierMap = params.getCarrierMap();
    frameTool = IEEE80211aFrame(params);
    cnn = HPA_CNN( ...
        'NFFT', params.FFTLength, ...
        'ModOrder', params.ModulationOrder, ...
        'NumInputChannels', params.NumCNNInputChannels, ...
        'MiniBatchSize', params.TrainingMiniBatchSize, ...
        'InitialLearnRate', 1e-3, ...
        'SymbolMSEWeight', 0.5);

    dataIdx = carrierMap.PayloadGlobalIdx;
    useGPU = params.UseGPUIfAvailable && canUseGPU();
    avgGrad = [];
    avgSqGrad = [];
    iteration = 0;
    startPhase = 1;
    probeReport = struct();

    resumePath = string(cfg.ResumeModelPath);
    if strlength(resumePath) > 0
        loaded = load(resumePath, 'net', 'trainingState', 'optimizerState');
        validate_resume_checkpoint(loaded);
        cnn.Net = loaded.net;
        if isfield(loaded, 'optimizerState')
            avgGrad = loaded.optimizerState.avgGrad;
            avgSqGrad = loaded.optimizerState.avgSqGrad;
            iteration = loaded.optimizerState.iteration;
        end
        if strcmp(requestedDirection, 'auto')
            direction = char(loaded.trainingState.curriculumDirection);
        else
            direction = requestedDirection;
        end
        if isempty(cfg.StartPhase)
            startPhase = loaded.trainingState.phaseNumber + 1;
        else
            startPhase = double(cfg.StartPhase);
        end
    else
        if ~isempty(cfg.StartPhase) && double(cfg.StartPhase) ~= 1
            error('StartPhase > 1 richiede ResumeModelPath.');
        end
        if strcmp(requestedDirection, 'auto')
            [direction, cnn.Net, probeReport] = select_direction( ...
                cnn, params, frameTool, carrierMap, dataIdx, useGPU, profile);
        else
            direction = requestedDirection;
        end
    end

    phases = build_curriculum(direction, profile);
    if startPhase < 1 || startPhase ~= round(startPhase) || ...
            startPhase > numel(phases)
        error('StartPhase deve essere un intero tra 1 e %d.', numel(phases));
    end

    fprintf("=== HPA CNN TRAINER | RAPP AM/AM ===\n");
    fprintf("Frame 802.11a: L-STF + L-LTF + %d simboli QPSK uncoded\n", ...
        params.NumOFDMSymbols);
    fprintf("CNN invariata: %s | input %d x %d\n", ...
        HPA_CNN.architectureVersion(), cnn.NumInputChannels, cnn.NFFT);
    fprintf("Contratto dati: %s | nessuna normalizzazione di potenza\n", ...
        model_data_contract());
    fprintf("Curriculum: %s | profilo: %s | esecuzione: %s\n", ...
        direction, profile, execution_description(useGPU));
    if strlength(resumePath) > 0
        fprintf("Ripresa: %s | fase %d\n", resumePath, startPhase);
    end

    lastCheckpoint = "";
    lastMetrics = struct();
    for phaseIdx = startPhase:numel(phases)
        phase = phases(phaseIdx);
        fprintf("\n--- Fase %d/%d: %s ---\n%s\n", phaseIdx, ...
            numel(phases), phase.name, describe_phase(phase));

        [cnn, avgGrad, avgSqGrad, iteration] = optimize_phase( ...
            cnn, params, frameTool, phase, carrierMap, dataIdx, useGPU, ...
            avgGrad, avgSqGrad, iteration, true);

        validationPhase = phase;
        validationPhase.numExamples = phase.numValidationExamples;
        [XVal, TVal] = generate_batch(cnn, params, frameTool, ...
            validationPhase, validationPhase.numExamples, carrierMap);
        XVal = cnn.prepareInputBatch(XVal, carrierMap.ActiveGlobalIdx);
        lastMetrics = evaluate_dataset(cnn, XVal, TVal, dataIdx, useGPU, ...
            phase.symbolMSEWeight);
        print_metrics("VALIDATION", lastMetrics);

        lastCheckpoint = save_phase_checkpoint(paths.Checkpoints, ...
            phaseIdx, phase, cnn, lastMetrics, carrierMap, avgGrad, ...
            avgSqGrad, iteration, params, direction, profile, probeReport);
        fprintf("Checkpoint: %s\n", lastCheckpoint);
    end

    summary = struct('Checkpoint', lastCheckpoint, ...
        'CurriculumDirection', string(direction), 'Profile', string(profile), ...
        'ValidationMetrics', lastMetrics, 'Probe', probeReport);
    fprintf("\nADDESTRAMENTO COMPLETATO.\n");
end

function [direction, winnerNet, report] = select_direction( ...
        cnn, params, frameTool, carrierMap, dataIdx, useGPU, profile)
    fprintf("Probe curriculum: confronto 0->25 dB e 25->0 dB...\n");
    directions = ["saturation-to-linear", "linear-to-saturation"];
    scores = inf(1, numel(directions));
    metricsByDirection = cell(1, numel(directions));
    nets = cell(1, numel(directions));

    if strcmp(profile, 'smoke')
        probeExamples = 32;
        validationExamples = 32;
    else
        probeExamples = 512;
        validationExamples = 512;
    end

    saturation = make_phase("PROBE_SAT", [0 3], [2 3.5], [15 35], ...
        probeExamples, 1, 5e-4, 0.5, 0);
    linear = make_phase("PROBE_LIN", [15 25], [2 3.5], [15 35], ...
        probeExamples, 1, 5e-4, 0.5, 0);
    validation = make_phase("PROBE_VAL", [-5 25], [1.5 4], [5 35], ...
        validationExamples, 1, 5e-4, 0.5, 0);
    validation.iboValuesDb = [-5 0 5 15 25];
    baseNet = cnn.Net;

    for candidateIdx = 1:numel(directions)
        candidate = clone_cnn(cnn, baseNet);
        if directions(candidateIdx) == "saturation-to-linear"
            order = {saturation, linear};
        else
            order = {linear, saturation};
        end
        avgGrad = [];
        avgSqGrad = [];
        iteration = 0;
        for stageIdx = 1:2
            rng(810 + double(order{stageIdx}.iboRangeDb(1) > 10), 'twister');
            [candidate, avgGrad, avgSqGrad, iteration] = optimize_phase( ...
                candidate, params, frameTool, order{stageIdx}, carrierMap, ...
                dataIdx, useGPU, avgGrad, avgSqGrad, iteration, false);
        end
        rng(899, 'twister');
        [XVal, TVal] = generate_batch(candidate, params, frameTool, ...
            validation, validation.numExamples, carrierMap);
        XVal = candidate.prepareInputBatch(XVal, carrierMap.ActiveGlobalIdx);
        metrics = evaluate_dataset(candidate, XVal, TVal, dataIdx, ...
            useGPU, validation.symbolMSEWeight);
        scores(candidateIdx) = metrics.loss;
        metricsByDirection{candidateIdx} = metrics;
        nets{candidateIdx} = candidate.Net;
        fprintf("  %-23s loss %.4f | BER %.3e\n", ...
            directions(candidateIdx), metrics.loss, metrics.ber);
    end

    [~, winnerIdx] = min(scores);
    direction = char(directions(winnerIdx));
    winnerNet = nets{winnerIdx};
    report = struct('directions', directions, 'scores', scores, ...
        'metrics', {metricsByDirection}, 'selected', string(direction));
    fprintf("Probe selezionato: %s\n", direction);
end

function phases = build_curriculum(direction, profile)
    if strcmp(direction, 'saturation-to-linear')
        iboRanges = [0 3; 3 8; 8 16; 16 25];
    else
        iboRanges = [16 25; 8 16; 3 8; 0 3];
    end
    snrRanges = [18 35; 12 35; 6 35; 0 35];
    smoothRanges = [2 3.5; 1.8 3.8; 1.5 4; 1.5 4];
    standardExamples = [6000 10000 14000 18000];
    standardEpochs = [4 5 6 7];
    learnRates = [1e-3 6e-4 3e-4 1.5e-4];
    symbolWeights = [0.5 0.6 0.8 1.0];

    phases = repmat(make_phase("", [0 0], [1 1], [0 0], ...
        1, 1, 1e-3, 0.5, 0), 1, 4);
    for idx = 1:4
        phases(idx) = make_phase(sprintf("IBO_%02d_%02d", ...
            iboRanges(idx, 1), iboRanges(idx, 2)), iboRanges(idx, :), ...
            smoothRanges(idx, :), snrRanges(idx, :), ...
            standardExamples(idx), standardEpochs(idx), ...
            learnRates(idx), symbolWeights(idx), 0);
    end

    phases(5) = make_phase("ROBUST_MIX", [-5 25], [1.5 4], [0 35], ...
        24000, 8, 7e-5, 1.0, 0.60);
    phases(5).hardIBORangeDb = [-5 3];
    phases(5).hardSmoothnessRange = [1.5 3];
    phases(5).hardSNRRangeDb = [8 35];
    phases(5).iboValuesDb = [-5 0 5 15 25];

    if strcmp(profile, 'smoke')
        for idx = 1:numel(phases)
            phases(idx).numExamples = 32;
            phases(idx).numValidationExamples = 32;
            phases(idx).epochs = 1;
        end
    end
end

function phase = make_phase(name, iboRangeDb, smoothnessRange, snrRangeDb, ...
        numExamples, epochs, learnRate, symbolMSEWeight, hardCaseProbability)
    phase = struct();
    phase.name = string(name);
    phase.iboRangeDb = double(iboRangeDb);
    phase.smoothnessRange = double(smoothnessRange);
    phase.snrRangeDb = double(snrRangeDb);
    phase.numExamples = double(numExamples);
    phase.numValidationExamples = min(4096, ...
        max(512, round(0.05 * numExamples)));
    phase.epochs = double(epochs);
    phase.learnRate = double(learnRate);
    phase.symbolMSEWeight = double(symbolMSEWeight);
    phase.hardCaseProbability = double(hardCaseProbability);
    phase.hardIBORangeDb = [-5 3];
    phase.hardSmoothnessRange = [1.5 3];
    phase.hardSNRRangeDb = [8 35];
    phase.iboValuesDb = [];
end

function [cnn, avgGrad, avgSqGrad, iteration] = optimize_phase( ...
        cnn, params, frameTool, phase, carrierMap, dataIdx, useGPU, ...
        avgGrad, avgSqGrad, iteration, verbose)
    iterationsPerEpoch = ceil(phase.numExamples / cnn.MiniBatchSize);
    for epoch = 1:phase.epochs
        totalLoss = 0;
        examplesSeen = 0;
        for batchIdx = 1:iterationsPerEpoch
            batchCount = min(cnn.MiniBatchSize, ...
                phase.numExamples - (batchIdx - 1) * cnn.MiniBatchSize);
            [X, T] = generate_batch(cnn, params, frameTool, phase, ...
                batchCount, carrierMap);
            X = cnn.prepareInputBatch(X, carrierMap.ActiveGlobalIdx);
            T = cnn.normalizeTargetShape(T);
            dlX = make_dlarray(X, useGPU);
            dlT = make_dlarray(T, useGPU);
            iteration = iteration + 1;
            [loss, gradients] = dlfeval(@HPA_CNN.modelGradients, ...
                cnn.Net, dlX, dlT, dataIdx, single(cnn.Constellation), ...
                phase.symbolMSEWeight);
            [cnn.Net, avgGrad, avgSqGrad] = adamupdate(cnn.Net, ...
                gradients, avgGrad, avgSqGrad, iteration, phase.learnRate);
            totalLoss = totalLoss + gather_scalar(loss) * batchCount;
            examplesSeen = examplesSeen + batchCount;
        end
        if verbose
            fprintf("Epoca %2d/%2d | loss %.4f\n", epoch, phase.epochs, ...
                totalLoss / examplesSeen);
        end
    end
end

function [X, T] = generate_batch(cnn, params, frameTool, phase, ...
        batchCount, carrierMap)
    X = zeros(cnn.NumInputChannels, cnn.NFFT, batchCount, 'single');
    labels = repmat(uint16(params.PilotClass), cnn.NFFT, batchCount);
    dataIdx = carrierMap.PayloadGlobalIdx;
    sampleIdx = 0;
    attempts = 0;
    maxAttempts = max(100, 20 * batchCount);

    while sampleIdx < batchCount
        attempts = attempts + 1;
        if attempts > maxAttempts
            error('Generazione batch fallita dopo %d tentativi.', maxAttempts);
        end
        [iboDb, smoothness, snrDb] = sample_operating_point(phase);
        localParams = params;
        localParams.SNR_dB = snrDb;
        [txWaveform, txFrame] = frameTool.createRandomFrame();
        try
            rxWaveform = apply_hpa_impairments(txWaveform, localParams, ...
                iboDb, smoothness);
            rxFrame = frameTool.receive(rxWaveform);
        catch
            continue;
        end

        symbolsPerFrame = min([params.NumTrainingSymbolsPerFrame, ...
            params.NumOFDMSymbols, batchCount - sampleIdx]);
        symbolOrder = randperm(params.NumOFDMSymbols, symbolsPerFrame);
        for symbolIdx = symbolOrder
            sampleIdx = sampleIdx + 1;
            X(:, :, sampleIdx) = build_ofdm_cnn_features( ...
                rxFrame.FullGrid(:, symbolIdx), carrierMap, ...
                txFrame.PilotSymbols(:, symbolIdx), cnn.NumInputChannels, ...
                rxFrame.NoiseVariance);
            dataLabels = cnn.symbolsToClasses( ...
                txFrame.PayloadSymbols(:, symbolIdx));
            fullLabels = repmat(uint16(params.PilotClass), cnn.NFFT, 1);
            fullLabels(dataIdx) = uint16(dataLabels);
            labels(:, sampleIdx) = fullLabels;
        end
    end
    T = cnn.classesToOneHot(labels);
end

function [iboDb, smoothness, snrDb] = sample_operating_point(phase)
    if rand() < phase.hardCaseProbability
        iboRange = phase.hardIBORangeDb;
        smoothnessRange = phase.hardSmoothnessRange;
        snrRange = phase.hardSNRRangeDb;
    else
        iboRange = phase.iboRangeDb;
        smoothnessRange = phase.smoothnessRange;
        snrRange = phase.snrRangeDb;
    end
    if ~isempty(phase.iboValuesDb) && rand() < 0.5
        iboDb = phase.iboValuesDb(randi(numel(phase.iboValuesDb)));
    else
        iboDb = sample_range(iboRange);
    end
    smoothness = sample_range(smoothnessRange);
    snrDb = sample_range(snrRange);
end

function value = sample_range(rangeValue)
    value = rangeValue(1) + rand() * (rangeValue(end) - rangeValue(1));
end

function metrics = evaluate_dataset(cnn, X, T, dataIdx, useGPU, weight)
    X = cnn.normalizeInputShape(X);
    T = cnn.normalizeTargetShape(T);
    numExamples = size(X, 3);
    numBatches = ceil(numExamples / cnn.MiniBatchSize);
    totalLoss = 0;
    bitErrors = 0;
    totalBits = 0;
    correctSymbols = 0;
    errorEnergy = 0;
    referenceEnergy = 0;

    for batchIdx = 1:numBatches
        firstIdx = (batchIdx - 1) * cnn.MiniBatchSize + 1;
        lastIdx = min(batchIdx * cnn.MiniBatchSize, numExamples);
        batchCount = lastIdx - firstIdx + 1;
        dlY = forward(cnn.Net, make_dlarray(X(:, :, firstIdx:lastIdx), useGPU));
        dlT = make_dlarray(T(:, :, firstIdx:lastIdx), useGPU);
        ce = HPA_CNN.crossEntropyLoss(dlY, dlT, dataIdx);
        symbolLoss = HPA_CNN.softSymbolMSE(dlY, dlT, dataIdx, ...
            single(cnn.Constellation));
        totalLoss = totalLoss + gather_scalar(ce + weight * symbolLoss) * batchCount;
        scores = normalize_score_shape(gather(extractdata(dlY)), cnn, batchCount);
        trueClasses = cnn.oneHotToClasses(T(:, :, firstIdx:lastIdx));
        predictedClasses = cnn.oneHotToClasses(scores);
        trueData = trueClasses(dataIdx, :);
        predictedData = predictedClasses(dataIdx, :);
        correctSymbols = correctSymbols + sum(trueData(:) == predictedData(:));
        trueBits = cnn.classesToBits(trueData);
        predictedBits = cnn.classesToBits(predictedData);
        bitErrors = bitErrors + sum(trueBits(:) ~= predictedBits(:));
        totalBits = totalBits + numel(trueBits);
        softSymbols = cnn.scoresToSymbols(scores);
        trueSymbols = cnn.classesToSymbols(trueData);
        errorEnergy = errorEnergy + sum(abs(softSymbols(dataIdx, :) - trueSymbols).^2, 'all');
        referenceEnergy = referenceEnergy + sum(abs(trueSymbols).^2, 'all');
    end

    metrics = struct('loss', totalLoss / numExamples, ...
        'accuracy', correctSymbols / (numExamples * numel(dataIdx)), ...
        'ber', bitErrors / totalBits, ...
        'nmse', errorEnergy / max(referenceEnergy, eps));
    metrics.evm = sqrt(metrics.nmse);
end

function print_metrics(label, metrics)
    fprintf("%s | loss %.4f | accuracy %.2f%% | BER %.3e | EVM %.2f%%\n", ...
        label, metrics.loss, 100 * metrics.accuracy, metrics.ber, ...
        100 * metrics.evm);
end

function savePath = save_phase_checkpoint(checkpointDir, phaseNumber, phase, ...
        cnn, metrics, carrierMap, avgGrad, avgSqGrad, iteration, params, ...
        direction, profile, probeReport)
    timestamp = string(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    net = gather_network(cnn.Net);
    optimizerState = struct('avgGrad', gather_state(avgGrad), ...
        'avgSqGrad', gather_state(avgSqGrad), 'iteration', iteration);
    trainingState = struct();
    trainingState.timestamp = timestamp;
    trainingState.networkArchitecture = HPA_CNN.architectureVersion();
    trainingState.dataContract = model_data_contract();
    trainingState.distortionModel = "RAPP_AMAM";
    trainingState.normalization = "none";
    trainingState.phaseNumber = phaseNumber;
    trainingState.phase = phase;
    trainingState.validationMetrics = metrics;
    trainingState.carrierMap = carrierMap;
    trainingState.parameters = params.refreshDerived();
    trainingState.curriculumDirection = string(direction);
    trainingState.profile = string(profile);
    trainingState.probe = probeReport;
    trainingState.referenceDOI = "10.1109/TCOMM.2003.809289";
    savePath = fullfile(checkpointDir, sprintf('ck_p%02d_%s.mat', ...
        phaseNumber, char(timestamp)));
    save(savePath, 'net', 'trainingState', 'optimizerState', ...
        'metrics', 'phase', '-v7.3');
end

function validate_resume_checkpoint(loaded)
    if ~isfield(loaded, 'net') || ~isfield(loaded, 'trainingState')
        error('Checkpoint privo di net o trainingState.');
    end
    state = loaded.trainingState;
    if string(state.networkArchitecture) ~= HPA_CNN.architectureVersion()
        error('Architettura CNN incompatibile.');
    end
    if ~isfield(state, 'dataContract') || ...
            string(state.dataContract) ~= model_data_contract()
        error('Checkpoint precedente alla rimozione della normalizzazione.');
    end
end

function cnnCopy = clone_cnn(cnn, net)
    cnnCopy = HPA_CNN('NFFT', cnn.NFFT, 'ModOrder', cnn.ModOrder, ...
        'NumInputChannels', cnn.NumInputChannels, ...
        'MiniBatchSize', cnn.MiniBatchSize, ...
        'InitialLearnRate', cnn.InitialLearnRate, ...
        'SymbolMSEWeight', cnn.SymbolMSEWeight);
    cnnCopy.Net = net;
end

function scores = normalize_score_shape(scores, cnn, batchCount)
    if ismatrix(scores)
        scores = reshape(scores, size(scores, 1), size(scores, 2), 1);
    end
    if size(scores, 1) ~= cnn.ModOrder
        error('Output CNN non valido.');
    end
    if size(scores, 2) == cnn.NFFT && size(scores, 3) == batchCount
        return;
    end
    if size(scores, 2) == batchCount && size(scores, 3) == cnn.NFFT
        scores = permute(scores, [1 3 2]);
        return;
    end
    error('Shape output CNN non valida: [%s].', num2str(size(scores)));
end

function dlX = make_dlarray(X, useGPU)
    X = single(X);
    if useGPU
        X = gpuArray(X);
    end
    dlX = dlarray(X, 'CTB');
end

function value = gather_scalar(dlValue)
    value = double(gather(extractdata(dlValue)));
end

function net = gather_network(net)
    try
        net = dlupdate(@gather, net);
    catch
    end
end

function state = gather_state(state)
    if isempty(state)
        return;
    end
    try
        state = dlupdate(@gather, state);
    catch
    end
end

function text = describe_phase(phase)
    text = sprintf(['IBO %.1f..%.1f dB | p %.1f..%.1f | SNR %.1f..%.1f dB ' ...
        '| hard %.0f%% | lambda %.2f | N/epoca %d | epoche %d | LR %.1e'], ...
        phase.iboRangeDb, phase.smoothnessRange, phase.snrRangeDb, ...
        100 * phase.hardCaseProbability, phase.symbolMSEWeight, ...
        phase.numExamples, phase.epochs, phase.learnRate);
end

function value = model_data_contract()
    value = "rapp_abs_drive_no_power_norm_v1";
end

function root = resolve_project_root(requestedRoot)
    root = string(requestedRoot);
    if strlength(root) == 0
        root = string(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    end
    root = char(root);
end

function text = execution_description(useGPU)
    if useGPU
        text = 'GPU';
    else
        text = 'CPU';
    end
end
