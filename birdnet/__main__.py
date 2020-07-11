
import argparse
import os

from birdnet import analyze
from birdnet import config as cfg


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--i', default='audio',
                        help='Path to input file or directory.')
    parser.add_argument('--o', default='',
                        help='Path to output directory. If not specified,'
                        'the input directory will be used.')
    parser.add_argument('--filetype', default='wav',
                        help='Filetype of soundscape recordings.'
                        'Defaults to \'wav\'.')
    parser.add_argument('--results', default='raven',
                        help='Output format of analysis results.'
                        'Values in [\'audacity\', \'raven\'].'
                        'Defaults to \'raven\'.')
    parser.add_argument('--lat', type=float, default=-1,
                        help='Recording location latitude. Set -1 to ignore.')
    parser.add_argument('--lon', type=float, default=-1,
                        help='Recording location longitude. Set -1 to ignore.')
    parser.add_argument('--week', type=int, default=-1,
                        help='Week of the year when the recordings were made.'
                        'Values in [1, 48]. Set -1 to ignore.')
    parser.add_argument('--overlap', type=float, default=0.0,
                        help='Overlap in seconds between extracted'
                        'spectrograms. Values in [0.0, 2.9].')
    parser.add_argument('--spp', type=int, default=1,
                        help='Combines probabilities of multiple spectrograms'
                        'to one prediction. Defaults to 1.')
    parser.add_argument('--sensitivity', type=float, default=1.0,
                        help='Sigmoid sensitivity; Higher values result in'
                        'lower sensitivity. Values in [0.25, 2.0].'
                        'Defaults to 1.0.')
    parser.add_argument('--min_conf', type=float, default=0.1,
                        help='Minimum confidence threshold.'
                        'Values in [0.01, 0.99]. Defaults to 0.1.')

    args = parser.parse_args()

    # Parse dataset
    dataset = analyze.parseTestSet(args.i, args.filetype)
    if len(dataset) > 0:

        # Load model
        test_function = analyze.loadModel()

        # Load eBird grid data
        analyze.loadGridData()

        # Adjust config
        cfg.DEPLOYMENT_LOCATION = (args.lat, args.lon)
        cfg.DEPLOYMENT_WEEK = args.week
        cfg.SPEC_OVERLAP = min(2.9, max(0.0, args.overlap))
        cfg.SPECS_PER_PREDICTION = max(1, args.spp)
        cfg.SENSITIVITY = max(min(-0.25, args.sensitivity * -1), -2.0)
        cfg.MIN_CONFIDENCE = min(0.99, max(0.01, args.min_conf))
        if len(args.o) == 0:
            if os.path.isfile(args.i):
                result_path = args.i.rsplit(os.sep, 1)[0]
            else:
                result_path = args.i
        else:
            result_path = args.o

        # Analyze dataset
        for s in dataset:
            analyze.process(s, dataset.index(s) + 1, result_path,
                            args.results, test_function)


if __name__ == '__main__':

    main()
