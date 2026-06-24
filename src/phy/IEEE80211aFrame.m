classdef IEEE80211aFrame < handle
    % Frame legacy 802.11a: L-STF, L-LTF e payload QPSK uncoded.

    properties (SetAccess = private)
        Params
        CarrierMap
        ShortTrainingField
        LongTrainingField
        Preamble
        LongTrainingSpectrum
        NumDataSymbols
        NumPayloadCarriers
        PreambleLength
        DataSymbolLength
        ChannelGuardLength
    end

    methods
        function obj = IEEE80211aFrame(params)
            params = params.refreshDerived();
            obj.Params = params;
            obj.CarrierMap = params.getCarrierMap();
            obj.NumDataSymbols = params.NumOFDMSymbols;
            obj.NumPayloadCarriers = numel(obj.CarrierMap.PayloadActiveIdx);
            obj.DataSymbolLength = params.FFTLength + params.CPLength;
            obj.ChannelGuardLength = params.FFTLength;

            [obj.ShortTrainingField, obj.LongTrainingField, ...
                obj.LongTrainingSpectrum] = obj.buildLegacyPreamble();
            obj.Preamble = [obj.ShortTrainingField; obj.LongTrainingField];
            obj.PreambleLength = numel(obj.Preamble);
        end

        function [txFrame, frame] = createRandomFrame(obj)
            bits = uint8(randi([0 1], 2, obj.NumPayloadCarriers, ...
                obj.NumDataSymbols));
            [txFrame, frame] = obj.createFrame(bits);
        end

        function [txFrame, frame] = createFrame(obj, payloadBits)
            expectedSize = [2, obj.NumPayloadCarriers, obj.NumDataSymbols];
            if ~isequal(size(payloadBits), expectedSize)
                error('payloadBits deve avere dimensione [%s].', ...
                    num2str(expectedSize));
            end
            if any(payloadBits(:) ~= 0 & payloadBits(:) ~= 1)
                error('payloadBits deve contenere solo 0 e 1.');
            end

            payloadSymbols = qammod(double(payloadBits(:)), ...
                obj.Params.ModulationOrder, 'gray', 'InputType', 'bit', ...
                'UnitAveragePower', true);
            payloadSymbols = reshape(payloadSymbols, ...
                obj.NumPayloadCarriers, obj.NumDataSymbols);

            pilotSymbols = obj.Params.getPilotSymbols(obj.NumDataSymbols);
            activeGrid = complex(zeros(obj.CarrierMap.NumActiveCarriers, ...
                obj.NumDataSymbols));
            activeGrid(obj.CarrierMap.PayloadActiveIdx, :) = payloadSymbols;
            activeGrid(obj.CarrierMap.PilotActiveIdx, :) = pilotSymbols;

            payloadWaveform = obj.modulateActiveGrid(activeGrid);
            % La coda evita che la latenza interna del canale tronchi
            % l'ultimo simbolo utile pur mantenendo invariata la sua API.
            txFrame = [obj.Preamble; payloadWaveform; ...
                complex(zeros(obj.ChannelGuardLength, 1))];

            frame = struct();
            frame.PayloadBits = uint8(payloadBits);
            frame.PayloadSymbols = payloadSymbols;
            frame.PilotSymbols = pilotSymbols;
            frame.ActiveGrid = activeGrid;
            frame.PayloadWaveform = payloadWaveform;
        end

        function rx = receive(obj, rxSignal)
            rxSignal = double(rxSignal(:));
            [frameStart, detectedPeak] = obj.synchronize(rxSignal);
            requiredLength = obj.PreambleLength + ...
                obj.NumDataSymbols * obj.DataSymbolLength;
            rxAligned = obj.extractWithZeroPadding(rxSignal, frameStart, requiredLength);

            ltfStart = numel(obj.ShortTrainingField) + 1;
            ltf = rxAligned(ltfStart:obj.PreambleLength);
            nfft = obj.Params.FFTLength;
            gi = nfft / 2;
            y1 = obj.demodulateSymbolWithoutCP(ltf(gi + (1:nfft)));
            y2 = obj.demodulateSymbolWithoutCP(ltf(gi + nfft + (1:nfft)));

            activeIdx = obj.CarrierMap.ActiveGlobalIdx;
            xLtf = obj.LongTrainingSpectrum(activeIdx);
            hEstimate = ((y1(activeIdx) + y2(activeIdx)) / 2) ./ xLtf;
            noiseVariance = mean(abs(y1(activeIdx) - y2(activeIdx)).^2) / 2;
            noiseVariance = max(real(noiseVariance), eps);

            payloadStart = obj.PreambleLength + 1;
            payload = rxAligned(payloadStart:end);
            [activeGrid, fullGrid] = obj.demodulatePayload(payload);
            pilotSymbols = obj.Params.getPilotSymbols(obj.NumDataSymbols);

            rx = struct();
            rx.FrameStart = frameStart;
            rx.DetectedPreamblePeak = detectedPeak;
            rx.SynchronizationMetric = obj.syncMetric(rxSignal, detectedPeak);
            rx.ActiveGrid = activeGrid;
            rx.FullGrid = fullGrid;
            rx.LTFEquivalentResponse = hEstimate(:);
            rx.NoiseVariance = noiseVariance;
            rx.PilotSymbols = pilotSymbols;
            rx.PilotPhaseTrackingEnabled = false;
        end

        function waveform = modulateActiveGrid(obj, activeGrid)
            if size(activeGrid, 1) ~= obj.CarrierMap.NumActiveCarriers
                error('activeGrid deve avere %d righe.', ...
                    obj.CarrierMap.NumActiveCarriers);
            end

            nfft = obj.Params.FFTLength;
            fullGrid = complex(zeros(nfft, size(activeGrid, 2)));
            fullGrid(obj.CarrierMap.ActiveGlobalIdx, :) = activeGrid;
            useful = ifft(ifftshift(fullGrid, 1), nfft, 1) * ...
                nfft / sqrt(obj.CarrierMap.NumActiveCarriers);
            withCP = [useful(end-obj.Params.CPLength+1:end, :); useful];
            waveform = withCP(:);
        end
    end

    methods (Access = private)
        function [stf, ltf, ltfSpectrum] = buildLegacyPreamble(obj)
            nfft = obj.Params.FFTLength;
            nactive = obj.CarrierMap.NumActiveCarriers;
            subcarrierAxis = obj.CarrierMap.GlobalSubcarrierIdx;

            shortSpectrum = complex(zeros(nfft, 1));
            shortSubcarriers = [-24 -20 -16 -12 -8 -4 4 8 12 16 20 24];
            shortPattern = [ ...
                1+1i, -1-1i, 1+1i, -1-1i, -1-1i, 1+1i, ...
                -1-1i, -1-1i, 1+1i, 1+1i, 1+1i, 1+1i].';
            [present, shortIdx] = ismember(shortSubcarriers, subcarrierAxis);
            if ~all(present)
                error('Mappa L-STF incompatibile con la griglia OFDM.');
            end
            shortSpectrum(shortIdx) = sqrt(13/6) * shortPattern;
            shortTime = ifft(ifftshift(shortSpectrum), nfft) * nfft / sqrt(nactive);
            stf = repmat(shortTime(1:16), 10, 1);

            ltf53 = [ ...
                1 1 -1 -1 1 1 -1 1 -1 1 1 1 1 1 1 -1 -1 1 ...
                1 -1 1 -1 1 1 1 1 0 1 -1 -1 1 1 -1 1 -1 1 ...
                -1 -1 -1 -1 -1 1 1 -1 -1 1 -1 1 -1 1 1 1 1].';
            ltfSpectrum = complex(zeros(nfft, 1));
            ltfAxis = -26:26;
            [present, ltfIdx] = ismember(ltfAxis, subcarrierAxis);
            if ~all(present)
                error('Mappa L-LTF incompatibile con la griglia OFDM.');
            end
            ltfSpectrum(ltfIdx) = ltf53;
            ltfTime = ifft(ifftshift(ltfSpectrum), nfft) * nfft / sqrt(nactive);
            ltf = [ltfTime(end-31:end); ltfTime; ltfTime];
        end

        function [frameStart, detectedPeak] = synchronize(obj, rxSignal)
            if numel(rxSignal) < obj.PreambleLength
                error('Segnale ricevuto piu corto del preambolo.');
            end
            matched = conv(rxSignal, flipud(conj(obj.Preamble)), 'valid');
            [~, detectedPeak] = max(abs(matched));

            % Il massimo del correlatore segue spesso il path piu forte.
            % Un piccolo anticipo mantiene la FFT dentro il prefisso ciclico
            % anche quando il primo path e piu debole.
            backoff = min(floor(obj.Params.CPLength / 4), detectedPeak - 1);
            frameStart = detectedPeak - backoff;
        end

        function metric = syncMetric(obj, rxSignal, frameStart)
            stopIdx = frameStart + obj.PreambleLength - 1;
            if frameStart < 1 || stopIdx > numel(rxSignal)
                metric = 0;
                return;
            end
            segment = rxSignal(frameStart:stopIdx);
            metric = abs(obj.Preamble' * segment)^2 / ...
                ((sum(abs(obj.Preamble).^2) * sum(abs(segment).^2)) + eps);
        end

        function extracted = extractWithZeroPadding(~, signal, startIdx, count)
            extracted = complex(zeros(count, 1));
            sourceStart = max(startIdx, 1);
            sourceEnd = min(startIdx + count - 1, numel(signal));
            if sourceEnd < sourceStart
                return;
            end
            targetStart = sourceStart - startIdx + 1;
            targetEnd = targetStart + sourceEnd - sourceStart;
            extracted(targetStart:targetEnd) = signal(sourceStart:sourceEnd);
        end

        function [activeGrid, fullGrid] = demodulatePayload(obj, waveform)
            symbolLength = obj.DataSymbolLength;
            expectedLength = obj.NumDataSymbols * symbolLength;
            if numel(waveform) < expectedLength
                waveform(end+1:expectedLength, 1) = 0;
            end
            symbols = reshape(waveform(1:expectedLength), ...
                symbolLength, obj.NumDataSymbols);
            useful = symbols(obj.Params.CPLength + (1:obj.Params.FFTLength), :);
            fullGrid = fftshift(fft(useful, obj.Params.FFTLength, 1), 1) * ...
                sqrt(obj.CarrierMap.NumActiveCarriers) / obj.Params.FFTLength;
            activeGrid = fullGrid(obj.CarrierMap.ActiveGlobalIdx, :);
        end

        function fullSymbol = demodulateSymbolWithoutCP(obj, waveform)
            fullSymbol = fftshift(fft(waveform, obj.Params.FFTLength), 1) * ...
                sqrt(obj.CarrierMap.NumActiveCarriers) / obj.Params.FFTLength;
        end
    end
end
