function X = build_ofdm_cnn_features( ...
        rxFullGrid, carrierMap, pilotSymbols, numInputChannels, noiseVariance)
% rxFullGrid: [64 x 1]. X: [numInputChannels x 64].
% DC e guard band mantengono lo spectral regrowth prodotto dall'HPA.

    if nargin < 4 || isempty(numInputChannels)
        numInputChannels = 10;
    end
    if nargin < 5 || isempty(noiseVariance)
        noiseVariance = 0;
    end
    if numInputChannels < 2 || numInputChannels > 10
        error('numInputChannels deve essere compreso tra 2 e 10.');
    end
    validate_carrier_map(carrierMap);

    if ~isequal(size(rxFullGrid), [carrierMap.NumSubcarriers, 1])
        error('rxFullGrid deve essere [%d x 1].', carrierMap.NumSubcarriers);
    end
    if ~isequal(size(pilotSymbols), [numel(carrierMap.PilotActiveIdx), 1])
        error('pilotSymbols deve essere [numPiloti x 1].');
    end

    rxGrid = rxFullGrid(carrierMap.ActiveGlobalIdx);
    [linearEqualized, pilotGain] = equalize_mmse_pilots( ...
        rxGrid, pilotSymbols, carrierMap, noiseVariance);
    activeIdx = carrierMap.ActiveGlobalIdx;
    pilotIdx = carrierMap.PilotActiveIdx;

    X = zeros(numInputChannels, carrierMap.NumSubcarriers, 'single');
    X(1, :) = single(real(rxFullGrid)).';
    X(2, :) = single(imag(rxFullGrid)).';

    if numInputChannels >= 4
        X(3, activeIdx) = single(real(linearEqualized)).';
        X(4, activeIdx) = single(imag(linearEqualized)).';
    end

    if numInputChannels >= 5
        X(5, activeIdx) = single(abs(pilotGain));
    end

    if numInputChannels >= 6
        pilotResidual = linearEqualized(pilotIdx) - pilotSymbols;
        pilotResidualRMS = sqrt(mean(abs(pilotResidual).^2));
        X(6, activeIdx) = single(pilotResidualRMS);
    end

    if numInputChannels >= 7
        X(7, activeIdx) = single(abs(linearEqualized)).';
    end

    if numInputChannels >= 8
        X(8, carrierMap.PilotGlobalIdx) = 1;
    end

    if numInputChannels >= 9
        X(9, activeIdx) = 1;
    end

    if numInputChannels >= 10
        coordinate = double(carrierMap.GlobalSubcarrierIdx);
        X(10, :) = single(coordinate ./ max(abs(coordinate)));
    end
end

function validate_carrier_map(carrierMap)
    requiredFields = {'NumSubcarriers', 'NumActiveCarriers', ...
        'ActiveGlobalIdx', 'PilotActiveIdx', 'PilotGlobalIdx'};
    for k = 1:numel(requiredFields)
        if ~isfield(carrierMap, requiredFields{k})
            error('carrierMap manca il campo %s.', requiredFields{k});
        end
    end
end
