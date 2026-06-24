classdef simulation_parameters

    properties
        %% Modulazione e frame IEEE 802.11a

        ModulationOrder = 4;
        FFTLength = 64;
        CPLength = 16;
        NumOFDMSymbols = 50;
        DataRateMbps = 12;
        UseFEC = false;
        NumTrainingSymbolsPerFrame = 12;

        NumGuardBandCarriers = [6; 5];
        InsertDCNull = true;
        NumPilots = 4;
        PilotSubcarrierIndices = [-21 -7 7 21];
        PilotBasePattern = [1; 1; 1; -1];
        PilotClass = 1;

        SubcarrierBandwidthHz = 312.5e3;
        SubcarrierSpacingHz = 312.5e3;
        SampleRate = 20e6;
        UsefulSymbolDuration = 3.2e-6;
        CPDuration = 0.8e-6;
        OFDMSymbolDuration = 4.0e-6;

        %% HPA Rapp memoryless (AM/AM)

        RappSmoothness = 2;
        HPAOutputSaturationLevel = 1;
        HPAInputBackOffDb = 0;
        TrainingIBORangeDb = [-5 25];
        TrainingIBOAnchorsDb = [0 5 15 25];
        TrainingSmoothnessRange = [1.5 4];
        TrainingSNRRangeDb = [0 35];
        TestIBODb = [-5 0 5 15 25];
        TestSNRDb = 0:2:34;

        % SNR campione dell'AWGN piatto applicato dopo l'HPA.
        SNR_dB = Inf;
        SNRDefinition = 'SNR';

        %% Dataset / esecuzione

        DatasetSize = 5e15;
        TrainingMiniBatchSize = 256;
        TestMiniBatchSize = 1024;
        UseGPUIfAvailable = false;
        NumCNNInputChannels = 10;

        %% Derivati

        NumDataCarriers
        NumPayloadCarriers
        BitsPerPayloadOFDMSymbol
    end

    methods
        function obj = simulation_parameters()
            obj = obj.refreshDerived();
        end

        function obj = refreshDerived(obj)
            if ~isempty(obj.PilotSubcarrierIndices)
                obj.NumPilots = numel(obj.PilotSubcarrierIndices);
            end
            obj.validateBaseParameters();

            obj.SubcarrierSpacingHz = obj.SubcarrierBandwidthHz;
            obj.SampleRate = obj.FFTLength * obj.SubcarrierBandwidthHz;
            obj.UsefulSymbolDuration = 1 / obj.SubcarrierSpacingHz;
            obj.CPDuration = obj.CPLength / obj.SampleRate;
            obj.OFDMSymbolDuration = obj.UsefulSymbolDuration + obj.CPDuration;

            obj.NumDataCarriers = obj.FFTLength ...
                - sum(obj.NumGuardBandCarriers) ...
                - double(obj.InsertDCNull);
            obj.NumPayloadCarriers = obj.NumDataCarriers - obj.NumPilots;
            obj.BitsPerPayloadOFDMSymbol = ...
                obj.NumPayloadCarriers * log2(obj.ModulationOrder);
        end

        function pilotIdx = getPilotIndices(obj, numCarriers)
            if nargin < 2 || isempty(numCarriers)
                numCarriers = obj.NumDataCarriers;
            end

            p = obj.refreshDerived();
            if p.NumPilots < 1 || p.NumPilots >= numCarriers
                error('Numero di piloti non valido per %d carrier.', numCarriers);
            end

            if numCarriers == p.NumDataCarriers
                activeSubcarrierIdx = p.getActiveSubcarrierIndices();
                [isPilot, pilotIdx] = ismember( ...
                    double(p.PilotSubcarrierIndices(:)).', activeSubcarrierIdx);
                if ~all(isPilot)
                    error('PilotSubcarrierIndices contiene toni non attivi.');
                end
                return;
            end

            pilotIdx = round(linspace(2, numCarriers - 1, p.NumPilots));
            if numel(unique(pilotIdx)) ~= p.NumPilots
                error('Impossibile posizionare %d piloti distinti.', p.NumPilots);
            end
        end

        function subcarrierIdx = getSubcarrierIndices(obj)
            p = obj.refreshDerived();
            subcarrierIdx = (-p.FFTLength/2):(p.FFTLength/2 - 1);
        end

        function activeSubcarrierIdx = getActiveSubcarrierIndices(obj)
            p = obj.refreshDerived();
            subcarrierIdx = p.getSubcarrierIndices();
            activeSubcarrierIdx = subcarrierIdx(p.getActiveCarrierIndices());
        end

        function pilotSymbols = getPilotSymbols(obj, numSymbols, symbolOffset)
            if nargin < 2 || isempty(numSymbols)
                numSymbols = obj.NumOFDMSymbols;
            end
            if nargin < 3 || isempty(symbolOffset)
                symbolOffset = 0;
            end
            if numSymbols < 1 || numSymbols ~= round(numSymbols)
                error('numSymbols deve essere un intero positivo.');
            end

            p = obj.refreshDerived();
            basePattern = double(p.PilotBasePattern(:));
            polarity = p.ieee80211aPilotPolaritySequence();
            seqIdx = mod(double(symbolOffset) + (0:numSymbols-1), ...
                numel(polarity)) + 1;
            pilotSymbols = complex(basePattern * polarity(seqIdx));
        end

        function activeIdx = getActiveCarrierIndices(obj)
            p = obj.refreshDerived();
            activeMask = true(p.FFTLength, 1);
            activeMask(1:p.NumGuardBandCarriers(1)) = false;
            if p.NumGuardBandCarriers(2) > 0
                activeMask(end-p.NumGuardBandCarriers(2)+1:end) = false;
            end
            if p.InsertDCNull
                activeMask(p.FFTLength/2 + 1) = false;
            end

            activeIdx = find(activeMask).';
            if numel(activeIdx) ~= p.NumDataCarriers
                error('Mappa carrier IEEE 802.11a non valida.');
            end
        end

        function carrierMap = getCarrierMap(obj)
            p = obj.refreshDerived();
            subcarrierIdx = p.getSubcarrierIndices();
            activeIdx = p.getActiveCarrierIndices();
            pilotActiveIdx = p.getPilotIndices(p.NumDataCarriers);
            payloadActiveIdx = setdiff(1:p.NumDataCarriers, pilotActiveIdx);

            carrierMap = struct();
            carrierMap.NumSubcarriers = p.FFTLength;
            carrierMap.NumActiveCarriers = p.NumDataCarriers;
            carrierMap.GlobalSubcarrierIdx = subcarrierIdx;
            carrierMap.ActiveGlobalIdx = activeIdx;
            carrierMap.ActiveSubcarrierIdx = subcarrierIdx(activeIdx);
            carrierMap.InactiveGlobalIdx = setdiff(1:p.FFTLength, activeIdx);
            carrierMap.PilotActiveIdx = pilotActiveIdx;
            carrierMap.PilotGlobalIdx = activeIdx(pilotActiveIdx);
            carrierMap.PilotSubcarrierIdx = ...
                subcarrierIdx(activeIdx(pilotActiveIdx));
            carrierMap.PayloadActiveIdx = payloadActiveIdx;
            carrierMap.PayloadGlobalIdx = activeIdx(payloadActiveIdx);
            carrierMap.PayloadSubcarrierIdx = ...
                subcarrierIdx(activeIdx(payloadActiveIdx));
        end
    end

    methods (Access = private)
        function validateBaseParameters(obj)
            if obj.ModulationOrder ~= 4
                error('Questo progetto e configurato solo per 4-QAM.');
            end
            if obj.DataRateMbps ~= 12 || obj.UseFEC
                error('Il payload deve restare 802.11a QPSK uncoded a 12 Mb/s.');
            end
            if obj.FFTLength ~= 64 || obj.CPLength ~= 16
                error('IEEE 802.11a richiede FFTLength=64 e CPLength=16.');
            end
            if ~isequal(double(obj.NumGuardBandCarriers(:)), [6; 5])
                error('IEEE 802.11a richiede NumGuardBandCarriers=[6;5].');
            end
            if ~obj.InsertDCNull
                error('IEEE 802.11a richiede InsertDCNull=true.');
            end
            if obj.SubcarrierBandwidthHz <= 0
                error('SubcarrierBandwidthHz deve essere positivo.');
            end
            if numel(obj.PilotBasePattern) ~= obj.NumPilots
                error('PilotBasePattern deve avere NumPilots elementi.');
            end
            if any(obj.PilotSubcarrierIndices == 0)
                error('I piloti non possono occupare la DC.');
            end
            if obj.NumTrainingSymbolsPerFrame < 1 || ...
                    obj.NumTrainingSymbolsPerFrame ~= ...
                    round(obj.NumTrainingSymbolsPerFrame)
                error('NumTrainingSymbolsPerFrame deve essere intero positivo.');
            end
            if obj.RappSmoothness <= 0
                error('RappSmoothness deve essere positivo.');
            end
            if obj.HPAOutputSaturationLevel <= 0
                error('HPAOutputSaturationLevel deve essere positivo.');
            end
            if numel(obj.TrainingIBORangeDb) ~= 2 || ...
                    obj.TrainingIBORangeDb(1) > obj.TrainingIBORangeDb(2)
                error('TrainingIBORangeDb deve essere [min max].');
            end
            if isempty(obj.TrainingIBOAnchorsDb) || ...
                    any(~isfinite(obj.TrainingIBOAnchorsDb))
                error('TrainingIBOAnchorsDb deve contenere valori finiti.');
            end
            if numel(obj.TrainingSmoothnessRange) ~= 2 || ...
                    obj.TrainingSmoothnessRange(1) <= 0 || ...
                    obj.TrainingSmoothnessRange(1) > ...
                    obj.TrainingSmoothnessRange(2)
                error('TrainingSmoothnessRange deve essere [min max] positivo.');
            end
            if numel(obj.TrainingSNRRangeDb) ~= 2 || ...
                    obj.TrainingSNRRangeDb(1) > obj.TrainingSNRRangeDb(2)
                error('TrainingSNRRangeDb deve essere [min max].');
            end
            if isempty(obj.TestIBODb) || any(~isfinite(obj.TestIBODb))
                error('TestIBODb deve contenere valori finiti.');
            end
            if isempty(obj.TestSNRDb) || any(diff(obj.TestSNRDb) <= 0)
                error('TestSNRDb deve essere strettamente crescente.');
            end
        end

        function polarity = ieee80211aPilotPolaritySequence(~)
            polarity = [ ...
                1 1 1 1 -1 -1 -1 1 -1 -1 -1 -1 1 1 -1 1 ...
                -1 -1 1 1 -1 1 1 -1 1 1 1 1 1 1 -1 1 ...
                1 1 -1 1 1 -1 -1 1 1 1 -1 1 -1 -1 -1 1 ...
                -1 1 -1 -1 1 -1 -1 1 1 1 1 1 -1 -1 1 1 ...
                -1 -1 1 -1 1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 ...
                -1 1 -1 -1 1 -1 1 1 1 1 -1 1 -1 1 -1 1 ...
                -1 -1 -1 -1 -1 1 -1 1 1 -1 1 -1 1 1 1 -1 ...
                -1 1 -1 -1 -1 1 1 1 -1 -1 -1 -1 -1 -1 -1];
        end
    end
end
