classdef HPA_CNN < handle
    properties
        NFFT (1,1) double = 64
        ModOrder (1,1) double = 4
        NumInputChannels (1,1) double = 10
        MiniBatchSize (1,1) double = 64
        MaxEpochs (1,1) double = 50
        InitialLearnRate (1,1) double = 1e-3
        SymbolMSEWeight (1,1) double = 0.5
        Constellation
        Net
        TrainingInfo
    end

    methods
        function obj = HPA_CNN(varargin)
            if mod(nargin, 2) ~= 0
                error('Gli argomenti devono essere coppie nome/valore.');
            end

            for k = 1:2:nargin
                obj.(varargin{k}) = varargin{k+1};
            end

            obj.validateModOrder();
            if obj.NumInputChannels < 2
                error('La CNN richiede almeno due canali input I/Q.');
            end
            if obj.SymbolMSEWeight < 0
                error('SymbolMSEWeight non puo essere negativo.');
            end

            obj.Constellation = obj.buildConstellation();
            obj.Net = dlnetwork(obj.buildLayerGraph());
        end

        function lgraph = buildLayerGraph(obj)
            lgraph = layerGraph([
                sequenceInputLayer(obj.NumInputChannels, ...
                    'Name', 'input', ...
                    'Normalization', 'none', ...
                    'MinLength', obj.NFFT)
                convolution1dLayer(1, 64, 'Padding', 'same', 'Name', 'stem')
                reluLayer('Name', 'stem_relu')
                ]);

            previousLayer = 'stem_relu';
            dilations = [1 2 4 8 1];
            for b = 1:numel(dilations)
                blockLayers = [
                    convolution1dLayer(9, 64, 'Padding', 'same', ...
                        'DilationFactor', dilations(b), ...
                        'Name', sprintf('res%d_conv1', b))
                    reluLayer('Name', sprintf('res%d_relu1', b))
                    convolution1dLayer(9, 64, 'Padding', 'same', ...
                        'DilationFactor', dilations(b), ...
                        'Name', sprintf('res%d_conv2', b))
                    additionLayer(2, 'Name', sprintf('res%d_add', b))
                    reluLayer('Name', sprintf('res%d_relu_out', b))
                    ];

                lgraph = addLayers(lgraph, blockLayers);
                lgraph = connectLayers(lgraph, previousLayer, sprintf('res%d_conv1', b));
                lgraph = connectLayers(lgraph, previousLayer, sprintf('res%d_add/in2', b));
                previousLayer = sprintf('res%d_relu_out', b);
            end

            headLayers = [
                convolution1dLayer(1, 48, 'Padding', 'same', 'Name', 'carrier_mixer')
                reluLayer('Name', 'head_relu')
                convolution1dLayer(1, obj.ModOrder, 'Padding', 'same', 'Name', 'class_scores')
                softmaxLayer('Name', 'softmax')
                ];

            lgraph = addLayers(lgraph, headLayers);
            lgraph = connectLayers(lgraph, previousLayer, 'carrier_mixer');
        end

        function X = normalizeInputShape(obj, X)
            % Input CNN: [NumInputChannels x NFFT x batch].
            if ~isreal(X)
                error('Input non valido: X deve contenere feature reali I/Q separate.');
            end

            if ismatrix(X)
                if size(X, 1) ~= obj.NumInputChannels || size(X, 2) ~= obj.NFFT
                    error('Input 2D non valido. Atteso [NumInputChannels x NFFT].');
                end
                X = reshape(X, obj.NumInputChannels, obj.NFFT, 1);
            elseif ndims(X) ~= 3
                error('Input non valido. Atteso [NumInputChannels x NFFT x batch].');
            end

            if size(X, 1) ~= obj.NumInputChannels || size(X, 2) ~= obj.NFFT
                error('Input non valido. Atteso [NumInputChannels x NFFT x batch].');
            end
        end

        function Xn = prepareInputBatch(obj, X, carrierIdx)
            X = obj.normalizeInputShape(X);

            if nargin >= 3 && ~isempty(carrierIdx)
                carrierIdx = double(carrierIdx(:)).';
                if any(carrierIdx < 1) || any(carrierIdx > obj.NFFT)
                    error('carrierIdx fuori range per NFFT=%d.', obj.NFFT);
                end
            end

            if any(~isfinite(X(:)))
                error('Le feature CNN contengono valori non finiti.');
            end

            % Nessuna AGC, standardizzazione o clipping: il livello assoluto
            % e parte dell'informazione necessaria a riconoscere l'IBO.
            Xn = single(X);
        end

        function Xn = normalizeInputBatch(obj, X, carrierIdx)
            % Alias retrocompatibile. Non effettua alcuna normalizzazione.
            if nargin < 3
                carrierIdx = [];
            end
            Xn = obj.prepareInputBatch(X, carrierIdx);
        end

        function T = normalizeTargetShape(obj, T)
            % Target: one-hot [M x NFFT x batch] oppure classi [NFFT x batch].
            if ismatrix(T)
                if size(T, 1) == obj.ModOrder && size(T, 2) == obj.NFFT && obj.looksLikeOneHot(T)
                    T = reshape(single(T), obj.ModOrder, obj.NFFT, 1);
                elseif size(T, 1) == obj.NFFT
                    T = obj.classesToOneHot(T);
                else
                    error('Target 2D non valido. Atteso [NFFT x batch] o [M x NFFT].');
                end
            elseif ndims(T) ~= 3
                error('Target non valido. Atteso [M x NFFT x batch].');
            end

            if size(T, 1) ~= obj.ModOrder || size(T, 2) ~= obj.NFFT
                error('Target one-hot non valido. Atteso [M x NFFT x batch].');
            end

            obj.validateOneHot(T);
            T = single(T);
        end

        function oneHot = classesToOneHot(obj, classes)
            % classes: [NFFT x batch], valori interi 1..M.
            if isvector(classes)
                classes = classes(:);
            end
            if ~ismatrix(classes)
                error('Class labels non validi. Atteso [NFFT x batch].');
            end

            classes = obj.validateClasses(classes);
            [nfft, nframes] = size(classes);
            oneHot = zeros(obj.ModOrder, nfft, nframes, 'single');

            classIdx = classes(:);
            carrierIdx = repmat((1:nfft).', nframes, 1);
            frameIdx = reshape(repmat(1:nframes, nfft, 1), [], 1);
            linearIdx = sub2ind([obj.ModOrder, nfft, nframes], classIdx, carrierIdx, frameIdx);
            oneHot(linearIdx) = 1;
        end

        function classes = oneHotToClasses(obj, oneHot)
            % oneHot: [M x NFFT x batch] oppure [M x NFFT].
            if ismatrix(oneHot)
                if size(oneHot, 1) ~= obj.ModOrder
                    error('One-hot non valido. La prima dimensione deve essere M.');
                end
                [~, idx] = max(oneHot, [], 1);
                classes = reshape(double(idx), size(oneHot, 2), 1);
            elseif ndims(oneHot) == 3
                if size(oneHot, 1) ~= obj.ModOrder
                    error('One-hot non valido. La prima dimensione deve essere M.');
                end
                [~, idx] = max(oneHot, [], 1);
                classes = reshape(double(idx), size(oneHot, 2), size(oneHot, 3));
            else
                error('One-hot non valido. Atteso [M x NFFT x batch].');
            end
        end

        function symbols = classesToSymbols(obj, classes)
            % Le classi interne sono 1..M; qammod usa indici 0..M-1.
            classes = obj.validateClasses(classes);
            qamIntegers = classes(:) - 1;
            symbols = qammod(qamIntegers, obj.ModOrder, 'gray', 'UnitAveragePower', true);
            symbols = reshape(symbols, size(classes));
        end

        function symbols = scoresToSymbols(obj, scores)
            % scores: [M x NFFT x batch], uscita softmax della rete.
            if ismatrix(scores)
                scores = reshape(scores, size(scores, 1), size(scores, 2), 1);
            end
            if ndims(scores) ~= 3 || size(scores, 1) ~= obj.ModOrder || ...
                    size(scores, 2) ~= obj.NFFT
                error('Score non validi. Atteso [M x NFFT x batch].');
            end
            symbols = sum(scores .* reshape(obj.Constellation, [], 1, 1), 1);
            symbols = reshape(symbols, obj.NFFT, size(scores, 3));
        end

        function classes = symbolsToClasses(obj, symbols)
            qamIntegers = qamdemod(symbols(:), obj.ModOrder, 'gray', ...
                'UnitAveragePower', true, 'OutputType', 'integer');
            classes = reshape(double(qamIntegers) + 1, size(symbols));
        end

        function bits = classesToBits(obj, classes)
            symbols = obj.classesToSymbols(classes);
            bits = obj.symbolsToBits(symbols);
        end

        function classes = bitsToClasses(obj, bits)
            symbols = obj.bitsToSymbols(bits);
            classes = obj.symbolsToClasses(symbols);
        end

        function symbols = bitsToSymbols(obj, bits)
            [bits, outSize] = obj.validateBits(bits);
            symbols = qammod(double(bits(:)), obj.ModOrder, 'gray', ...
                'InputType', 'bit', 'UnitAveragePower', true);
            symbols = reshape(symbols, outSize);
        end

        function bits = symbolsToBits(obj, symbols)
            bitsColumn = qamdemod(symbols(:), obj.ModOrder, 'gray', ...
                'UnitAveragePower', true, 'OutputType', 'bit');
            bits = reshape(uint8(bitsColumn(:)), [obj.bitsPerSymbol(), size(symbols)]);
        end

        function ber = computeBER(obj, trueClasses, predClasses)
            trueClasses = obj.validateClasses(trueClasses);
            predClasses = obj.validateClasses(predClasses);
            if ~isequal(size(trueClasses), size(predClasses))
                error('trueClasses e predClasses devono avere la stessa dimensione.');
            end

            trueBits = obj.classesToBits(trueClasses);
            predBits = obj.classesToBits(predClasses);
            ber = mean(trueBits(:) ~= predBits(:));
        end

        function evm = computeEVM(~, trueSymbols, predSymbols)
            num = sum(abs(trueSymbols - predSymbols).^2, 'all');
            den = sum(abs(trueSymbols).^2, 'all') + eps;
            evm = sqrt(num / den);
        end

        function mse = computeMSEOneHot(obj, trueClasses, predClasses)
            trueOneHot = obj.classesToOneHot(trueClasses);
            predOneHot = obj.classesToOneHot(predClasses);
            mse = mean((trueOneHot(:) - predOneHot(:)).^2);
        end
    end

    methods (Static)
        function version = architectureVersion()
            version = "carrier_resnet_v1_rapp_soft_symbol_loss";
        end

        function Y = reshapeToClassTime(X, M, NFFT)
            Y = reshape(X, [M, NFFT, size(X, 2)]);
        end

        function loss = crossEntropyLoss(dlY, dlT, carrierIdx)
            if nargin >= 3 && ~isempty(carrierIdx)
                carrierIdx = double(carrierIdx(:)).';
                dlY = HPA_CNN.selectCarrierDimension(dlY, carrierIdx);
                dlT = HPA_CNN.selectCarrierDimension(dlT, carrierIdx);
            end

            carrierDim = HPA_CNN.dimensionByLabel(dlT, 'T', 2);
            batchDim = HPA_CNN.dimensionByLabel(dlT, 'B', 3);
            lossTerms = -dlT .* log(dlY + single(1e-8));
            loss = sum(lossTerms, 'all') / (size(dlT, carrierDim) * size(dlT, batchDim));
        end

        function [loss, gradients, crossEntropy, symbolMSE] = modelGradients( ...
                net, dlX, dlT, carrierIdx, constellation, symbolMSEWeight)
            if nargin < 4
                carrierIdx = [];
            end
            if nargin < 5 || isempty(constellation)
                error('La costellazione deve essere passata a modelGradients.');
            end
            if nargin < 6 || isempty(symbolMSEWeight)
                symbolMSEWeight = 0.5;
            end

            dlY = forward(net, dlX);
            crossEntropy = HPA_CNN.crossEntropyLoss(dlY, dlT, carrierIdx);
            symbolMSE = HPA_CNN.softSymbolMSE( ...
                dlY, dlT, carrierIdx, constellation);
            loss = crossEntropy + symbolMSEWeight * symbolMSE;
            gradients = dlgradient(loss, net.Learnables);
        end

        function loss = softSymbolMSE(dlY, dlT, carrierIdx, constellation)
            if nargin >= 3 && ~isempty(carrierIdx)
                dlY = HPA_CNN.selectCarrierDimension(dlY, carrierIdx);
                dlT = HPA_CNN.selectCarrierDimension(dlT, carrierIdx);
            end

            constellation = constellation(:);
            cReal = reshape(single(real(constellation)), [], 1, 1);
            cImag = reshape(single(imag(constellation)), [], 1, 1);
            predReal = sum(dlY .* cReal, 1);
            predImag = sum(dlY .* cImag, 1);
            trueReal = sum(dlT .* cReal, 1);
            trueImag = sum(dlT .* cImag, 1);
            loss = mean((predReal - trueReal).^2 + ...
                (predImag - trueImag).^2, 'all');
        end

        function dlX = selectCarrierDimension(dlX, carrierIdx)
            carrierDim = HPA_CNN.dimensionByLabel(dlX, 'T', 2);
            subs = repmat({':'}, 1, max(ndims(stripdims(dlX)), carrierDim));
            subs{carrierDim} = carrierIdx;
            dlX = dlX(subs{:});
        end

        function dim = dimensionByLabel(dlX, label, defaultDim)
            dim = defaultDim;
            try
                fmt = char(dims(dlX));
                idx = find(fmt == label, 1);
                if ~isempty(idx)
                    dim = idx;
                end
            catch
            end
        end
    end

    methods (Access = private)
        function constellation = buildConstellation(obj)
            qamIntegers = (0:obj.ModOrder-1).';
            constellation = qammod(qamIntegers, obj.ModOrder, 'gray', ...
                'UnitAveragePower', true);
            constellation = constellation(:);
        end

        function validateModOrder(obj)
            k = log2(obj.ModOrder);
            if obj.ModOrder < 2 || abs(k - round(k)) > eps
                error('ModOrder deve essere una potenza di due.');
            end
        end

        function k = bitsPerSymbol(obj)
            k = round(log2(obj.ModOrder));
        end

        function classes = validateClasses(obj, classes)
            if isempty(classes)
                error('Class labels vuoti.');
            end
            if any(~isfinite(double(classes(:))))
                error('Classi non finite.');
            end
            if any(abs(double(classes(:)) - round(double(classes(:)))) > eps)
                error('Le classi devono essere intere.');
            end
            if any(classes(:) < 1) || any(classes(:) > obj.ModOrder)
                error('Classi fuori range. Devono essere tra 1 e M.');
            end
            classes = double(classes);
        end

        function [bits, outSize] = validateBits(obj, bits)
            if isempty(bits)
                error('Bit vuoti.');
            end

            k = obj.bitsPerSymbol();
            if isvector(bits) && size(bits, 1) ~= k
                if mod(numel(bits), k) ~= 0
                    error('Numero di bit non divisibile per log2(M).');
                end
                bits = reshape(bits, k, []);
            end

            if size(bits, 1) ~= k
                error('Bit non validi. Atteso [log2(M) x NFFT x batch].');
            end
            if any(bits(:) ~= 0 & bits(:) ~= 1)
                error('I bit devono essere 0 o 1.');
            end

            bits = uint8(bits);
            sz = size(bits);
            if ismatrix(bits)
                outSize = [sz(2), 1];
            else
                outSize = sz(2:end);
            end
        end

        function validateOneHot(~, oneHot)
            if any(oneHot(:) < -1e-6) || any(oneHot(:) > 1 + 1e-6)
                error('Target one-hot non valido: valori fuori [0, 1].');
            end

            sums = sum(oneHot, 1);
            if any(abs(sums(:) - 1) > 1e-4)
                error('Target one-hot non valido: ogni portante deve avere somma 1.');
            end
        end

        function tf = looksLikeOneHot(~, T)
            tf = all(abs(T(:) - round(T(:))) < 1e-6) && ...
                all(T(:) >= 0) && all(T(:) <= 1) && ...
                all(abs(sum(T, 1) - 1) < 1e-4);
        end
    end
end
