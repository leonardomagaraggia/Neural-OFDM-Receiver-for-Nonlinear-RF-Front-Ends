function smoke_test
%SMOKE_TEST Verifica rapida, senza addestramento.
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(fullfile(projectRoot, 'src')));
    rng(41, 'twister');

    params = simulation_parameters().refreshDerived();
    map = params.getCarrierMap();
    assert(map.NumActiveCarriers == 52);
    assert(numel(map.PayloadActiveIdx) == 48);
    assert(isequal(map.PilotSubcarrierIdx, [-21 -7 7 21]));

    cnn = HPA_CNN('NFFT', 64, 'ModOrder', 4, 'NumInputChannels', 10);
    raw = single(reshape(linspace(-12, 12, 640), 10, 64));
    prepared = cnn.prepareInputBatch(raw, map.ActiveGlobalIdx);
    assert(isequal(prepared, reshape(raw, 10, 64, 1)));

    probe = complex([0.1 0.4 0.8 1.2 2.0].', 0.2);
    [~, infoA] = non_linearity(probe, 2, 1, 0);
    [~, infoB] = non_linearity(2 * probe, 2, 1, 0);
    assert(abs(infoA.DriveScale - infoB.DriveScale) < eps);
    assert(infoB.InputPowerBeforeDrive > 3.9 * infoA.InputPowerBeforeDrive);

    [~, satLow] = non_linearity(probe, 2, 1, -5);
    [~, satMid] = non_linearity(probe, 2, 1, 0);
    [~, satHigh] = non_linearity(probe, 2, 1, 25);
    assert(satLow.SaturatedSampleFraction >= satMid.SaturatedSampleFraction);
    assert(satMid.SaturatedSampleFraction >= satHigh.SaturatedSampleFraction);

    params.SNR_dB = 20;
    frameTool = IEEE80211aFrame(params);
    [txWaveform, txFrame] = frameTool.createRandomFrame();
    [rxWaveform, hpaInfo] = apply_hpa_impairments(txWaveform, params, 0, 2);
    rxFrame = frameTool.receive(rxWaveform);
    [mmseGrid, channelEstimate] = equalize_mmse_pilots( ...
        rxFrame.ActiveGrid, txFrame.PilotSymbols, map, rxFrame.NoiseVariance);
    assert(isequal(size(mmseGrid), size(rxFrame.ActiveGrid)));
    assert(all(isfinite(channelEstimate), 'all'));
    X = build_frame_cnn_batch(rxFrame, txFrame, map, 10);
    assert(isequal(size(X), [10 64 params.NumOFDMSymbols]));
    assert(all(isfinite(X), 'all'));
    assert(hpaInfo.InputBackOffDb == 0);
    fprintf('SMOKE TEST OK | 802.11a, Rapp, MMSE e feature verificati.\n');
end
