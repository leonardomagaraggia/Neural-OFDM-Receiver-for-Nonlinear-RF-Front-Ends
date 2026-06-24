function [equalizedGrid, channelEstimate, mmseWeight] = ...
        equalize_mmse_pilots(rxGrid, pilotSymbols, carrierMap, noiseVariance)
% Baseline MMSE scalare per simbolo stimata dai quattro piloti 802.11a.

    if nargin < 4 || isempty(noiseVariance)
        noiseVariance = 0;
    end

    if size(rxGrid, 1) ~= carrierMap.NumActiveCarriers
        error('rxGrid deve avere una riga per carrier attivo.');
    end
    if size(pilotSymbols, 1) ~= numel(carrierMap.PilotActiveIdx) || ...
            size(pilotSymbols, 2) ~= size(rxGrid, 2)
        error('pilotSymbols deve essere [numPiloti x numSimboli].');
    end

    receivedPilots = rxGrid(carrierMap.PilotActiveIdx, :);
    numerator = sum(receivedPilots .* conj(pilotSymbols), 1);
    denominator = sum(abs(pilotSymbols).^2, 1) + eps;
    channelEstimate = numerator ./ denominator;

    noiseVariance = double(noiseVariance);
    if isscalar(noiseVariance)
        noiseVariance = repmat(noiseVariance, 1, size(rxGrid, 2));
    end
    if ~isequal(size(noiseVariance), [1, size(rxGrid, 2)]) || ...
            any(~isfinite(noiseVariance)) || any(noiseVariance < 0)
        error('noiseVariance deve essere scalare o [1 x numSimboli] non negativo.');
    end

    pilotPower = mean(abs(pilotSymbols).^2, 1);
    regularizer = noiseVariance ./ max(pilotPower, eps);
    mmseWeight = conj(channelEstimate) ./ ...
        (abs(channelEstimate).^2 + regularizer + eps);
    equalizedGrid = rxGrid .* mmseWeight;
end
