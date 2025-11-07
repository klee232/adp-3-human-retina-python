% Created by Kuan-Min Lee
% Created date: Oct. 20th, 2025
% All rights reserved to Leelab.ai

% Brief User Introducttion:
% The following code is created to convert 1-bit CCITT TIFFs to uncompressed uint8 TIFFs.

% Input Parameter: 
% inputDir (input directory)
% outputDir (output directory)

% note that, this function is intended to be created for more easy usage
% for pyhton


function convert_ccitt_to_uint8(inputDir, outputDir)
    % Ensure folders are valid
    if nargin < 2
        error('Usage: convert_ccitt_to_uint8(inputDir, outputDir)');
    end
    if ~isfolder(inputDir)
        error('Input folder not found: %s', inputDir);
    end
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    % Get all .tif files in the input directory
    files = dir(fullfile(inputDir, '*.tif'));
    if isempty(files)
        warning('No .tif files found in %s', inputDir);
        return;
    end

    % Loop through each file
    for k = 1:numel(files)
        inFile = fullfile(inputDir, files(k).name);
        outFile = fullfile(outputDir, files(k).name);

        try
            % Read the image (MATLAB auto-decompresses CCITT)
            I = imread(inFile);

            % Convert logical or 1-bit to uint8 (0/255)
            if islogical(I)
                I8 = uint8(I) * 255;
            elseif isa(I, 'uint8')
                I8 = I; % already ok
            else
                I8 = im2uint8(I); % fallback
            end

            % Save uncompressed TIFF
            imwrite(I8, outFile, 'Compression', 'none');
            fprintf('✔ Converted: %s\n', files(k).name);

        catch ME
            fprintf('⚠️  Failed: %s (%s)\n', files(k).name, ME.message);
        end
    end

    fprintf('✅ Done. Converted %d files to %s\n', numel(files), outputDir);
end