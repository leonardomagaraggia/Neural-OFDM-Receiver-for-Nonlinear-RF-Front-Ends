function logPath = write_test_log(results, chartInfo, logDir)
%WRITE_TEST_LOG Scrive un record JSON completo codificato Base64.
    if ~exist(logDir, 'dir')
        mkdir(logDir);
    end
    record = struct();
    record.schema = "HPA_CNN_TEST_LOG_V1";
    record.timestamp = results.Timestamp;
    record.channelModel = results.ChannelModel;
    record.dataContract = results.DataContract;
    record.referenceDOI = "10.1109/TCOMM.2003.809289";
    record.test = results.TestConfig;
    record.test.IBODb = results.IBODb;
    record.test.SNRDb = results.SNRDb;
    record.phy = public_parameters(results.Parameters);
    record.modelPath = results.ModelPath;
    record.metrics = struct('methodNames', results.MethodNames, ...
        'errors', results.Errors, 'totalBits', results.TotalBits, ...
        'numFrames', results.NumFrames, 'BER', results.BER, ...
        'saturatedSampleFraction', results.SaturatedSampleFraction, ...
        'actualIBODb', results.ActualIBODb, ...
        'cnnBetterFraction', results.CNNBetterFraction, ...
        'cnnNotWorseFraction', results.CNNNotWorseFraction, ...
        'zeroErrorConvention', results.ZeroErrorConvention);
    record.chart = rmfield_if_present(chartInfo, 'Figure');
    jsonText = jsonencode(record);
    encoded = base64_encode_utf8(jsonText);
    logPath = string(fullfile(logDir, sprintf('lg_%s.txt', ...
        char(results.Timestamp))));
    fileId = fopen(logPath, 'w');
    if fileId < 0
        error('Impossibile creare il log %s.', logPath);
    end
    cleaner = onCleanup(@() fclose(fileId));
    fwrite(fileId, encoded, 'char');
    fwrite(fileId, newline, 'char');
end

function output = public_parameters(params)
    output = struct();
    names = properties(params);
    for idx = 1:numel(names)
        value = params.(names{idx});
        if isnumeric(value) || islogical(value) || ischar(value) || isstring(value)
            output.(names{idx}) = value;
        end
    end
end

function value = rmfield_if_present(value, fieldName)
    if isstruct(value) && isfield(value, fieldName)
        value = rmfield(value, fieldName);
    end
end

function encoded = base64_encode_utf8(text)
    bytes = double(unicode2native(char(text), 'UTF-8'));
    bytes = bytes(:).';
    alphabet = ['ABCDEFGHIJKLMNOPQRSTUVWXYZ' ...
        'abcdefghijklmnopqrstuvwxyz0123456789+/'];
    padding = mod(3 - mod(numel(bytes), 3), 3);
    bytes = [bytes zeros(1, padding)];
    bytes = reshape(bytes, 3, []);
    indices = [floor(bytes(1, :) / 4); ...
        mod(bytes(1, :), 4) * 16 + floor(bytes(2, :) / 16); ...
        mod(bytes(2, :), 16) * 4 + floor(bytes(3, :) / 64); ...
        mod(bytes(3, :), 64)] + 1;
    encoded = reshape(alphabet(indices), 1, []);
    if padding > 0
        encoded(end-padding+1:end) = '=';
    end
end
