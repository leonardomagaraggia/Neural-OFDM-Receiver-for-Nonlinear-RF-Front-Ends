function [rxSignal, info] = apply_hpa_impairments( ...
        txSignal, params, inputBackOffDb, smoothness)
% Canale piatto: HPA Rapp AM/AM seguito da AWGN (h = 1).

    params = params.refreshDerived();
    if nargin < 3 || isempty(inputBackOffDb)
        inputBackOffDb = params.HPAInputBackOffDb;
    end
    if nargin < 4 || isempty(smoothness)
        smoothness = params.RappSmoothness;
    end
    [distortedSignal, hpaInfo] = non_linearity( ...
        txSignal, smoothness, params.HPAOutputSaturationLevel, ...
        inputBackOffDb);

    signalMask = abs(distortedSignal(:)) > sqrt(eps);
    if any(signalMask)
        signalPower = mean(abs(distortedSignal(signalMask)).^2);
    else
        signalPower = 0;
    end

    if isfinite(params.SNR_dB) && signalPower > 0
        sampleSnrDb = params.SNR_dB;
        if strcmpi(char(params.SNRDefinition), 'EbNo')
            bitsPerTimeSample = log2(params.ModulationOrder) * ...
                params.NumPayloadCarriers / ...
                (params.FFTLength + params.CPLength);
            sampleSnrDb = sampleSnrDb + 10 * log10(bitsPerTimeSample);
        end
        noiseVariance = signalPower / 10^(sampleSnrDb / 10);
        noise = sqrt(noiseVariance / 2) .* ...
            (randn(size(distortedSignal)) + 1i * randn(size(distortedSignal)));
        rxSignal = distortedSignal + noise;
    else
        sampleSnrDb = Inf;
        noiseVariance = 0;
        rxSignal = distortedSignal;
    end

    info = hpaInfo;
    info.NoiseVarianceTime = noiseVariance;
    info.SampleSNRdB = sampleSnrDb;
end
