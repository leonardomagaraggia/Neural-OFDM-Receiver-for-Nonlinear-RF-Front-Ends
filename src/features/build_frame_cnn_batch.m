function X = build_frame_cnn_batch(rxFrame, txFrame, carrierMap, numInputChannels)
% Costruisce un esempio CNN per ogni simbolo payload dello stesso frame.

    numSymbols = size(rxFrame.ActiveGrid, 2);
    X = zeros(numInputChannels, carrierMap.NumSubcarriers, numSymbols, 'single');

    for symbolIdx = 1:numSymbols
        X(:, :, symbolIdx) = build_ofdm_cnn_features( ...
            rxFrame.FullGrid(:, symbolIdx), carrierMap, ...
            txFrame.PilotSymbols(:, symbolIdx), numInputChannels, ...
            rxFrame.NoiseVariance);
    end
end
