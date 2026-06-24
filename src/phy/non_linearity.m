function [outSignal, info] = non_linearity( ...
        inSignal, smoothness, saturationLevel, inputBackOffDb)
% Rapp AM/AM memoryless con drive IBO assoluto, senza AGC per-frame.

    if nargin < 4 || isempty(inputBackOffDb)
        inputBackOffDb = 0;
    end
    if nargin < 3 || isempty(saturationLevel)
        saturationLevel = 1;
    end
    if nargin < 2 || isempty(smoothness)
        smoothness = 2;
    end
    if smoothness <= 0 || saturationLevel <= 0
        error('Smoothness e saturationLevel devono essere positivi.');
    end
    if ~isscalar(inputBackOffDb) || ~isfinite(inputBackOffDb)
        error('inputBackOffDb deve essere uno scalare finito.');
    end

    signalMask = abs(inSignal(:)) > sqrt(eps);
    if ~any(signalMask)
        outSignal = inSignal;
        info = empty_info(inputBackOffDb, smoothness, saturationLevel);
        return;
    end

    % Il modulatore 802.11a produce un waveform nominalmente a potenza unitaria.
    % L'IBO agisce quindi come drive fisso. Non viene misurata la potenza del
    % singolo frame per riscalarlo e l'uscita non viene compensata.
    driveScale = saturationLevel * 10^(-inputBackOffDb / 20);
    drivenSignal = driveScale .* inSignal;
    normalizedAmplitude = abs(drivenSignal) ./ saturationLevel;
    compression = (1 + normalizedAmplitude.^(2 * smoothness)) ...
        .^ (-1 / (2 * smoothness));
    outSignal = drivenSignal .* compression;

    activeCompression = compression(signalMask);
    inputPower = mean(abs(drivenSignal(signalMask)).^2);
    outputPower = mean(abs(outSignal(signalMask)).^2);
    info = struct();
    info.Model = "RAPP_AMAM";
    info.InputBackOffDb = inputBackOffDb;
    info.ActualInputBackOffDb = 10 * log10( ...
        saturationLevel^2 / max(inputPower, eps));
    info.Smoothness = smoothness;
    info.SaturationLevel = saturationLevel;
    info.DriveScale = driveScale;
    info.InputPowerBeforeDrive = mean(abs(inSignal(signalMask)).^2);
    info.DrivenInputPower = inputPower;
    info.OutputPower = outputPower;
    info.GainCompressionDb = 10 * log10(outputPower / max(inputPower, eps));
    info.SaturatedSampleFraction = mean(normalizedAmplitude(signalMask) >= 1);
    info.MeanAmplitudeGain = mean(activeCompression);
    info.InputPAPRDb = papr_db(inSignal(signalMask));
    info.OutputPAPRDb = papr_db(outSignal(signalMask));
end

function info = empty_info(inputBackOffDb, smoothness, saturationLevel)
    info = struct( ...
        'Model', "RAPP_AMAM", ...
        'InputBackOffDb', inputBackOffDb, ...
        'ActualInputBackOffDb', inputBackOffDb, ...
        'Smoothness', smoothness, ...
        'SaturationLevel', saturationLevel, ...
        'DriveScale', saturationLevel * 10^(-inputBackOffDb / 20), ...
        'InputPowerBeforeDrive', 0, ...
        'DrivenInputPower', 0, ...
        'OutputPower', 0, ...
        'GainCompressionDb', 0, ...
        'SaturatedSampleFraction', 0, ...
        'MeanAmplitudeGain', 1, ...
        'InputPAPRDb', 0, ...
        'OutputPAPRDb', 0);
end

function value = papr_db(signal)
    power = abs(signal).^2;
    value = 10 * log10(max(power) / (mean(power) + eps));
end
