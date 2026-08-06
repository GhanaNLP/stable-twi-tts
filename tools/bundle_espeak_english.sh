#!/usr/bin/env bash
# Build espeak-ng and extract the minimum needed to phonemise English on a device.
#
# A full espeak-ng data directory is ~31 MB because it carries 130 languages. English needs
# seven data files plus the shared library, ~1.6 MB in total, and the awkward part is knowing
# which: the compiled data lives in the *build* tree rather than the source tree, and the
# language file sits under a family directory (`gmw`, West Germanic) rather than at the top.
#
# Why link this instead of shipping the precomputed lexicon: unlimited vocabulary. The lexicon
# is smaller but silently drops any word it does not contain. Use the lexicon only where native
# linking is impossible.
#
# Licence: espeak-ng is GPL-3.0. Linking it into an application carries obligations that
# shipping a generated lexicon does not. Worth deciding deliberately rather than by default.
set -euo pipefail

OUT="${1:-espeak-english}"
JOBS="${JOBS:-8}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "building espeak-ng in $WORK (a few minutes)"
git clone -q --depth 1 https://github.com/espeak-ng/espeak-ng.git "$WORK/espeak-ng"
cd "$WORK/espeak-ng"
cmake -B build -DUSE_MBROLA=OFF -DUSE_LIBSONIC=OFF -DUSE_LIBPCAUDIO=OFF \
      -DBUILD_SHARED_LIBS=ON > /dev/null
cmake --build build -j"$JOBS" > /dev/null
cd - > /dev/null

mkdir -p "$OUT/espeak-ng-data/lang/gmw"
D="$WORK/espeak-ng/build/espeak-ng-data"

# phondata holds the acoustic data every language shares; en_dict is English's rules and
# exceptions compiled together. Omitting either leaves espeak unable to start.
for f in phontab phonindex phondata phondata-manifest intonations en_dict; do
    cp "$D/$f" "$OUT/espeak-ng-data/$f"
done
cp "$D/lang/gmw/en" "$OUT/espeak-ng-data/lang/gmw/en"
# -a preserves the symlinks: libespeak-ng.so and .so.1 point at the real .so.1.x.y, and a
# plain cp dereferences all three into full copies, nearly doubling the bundle.
cp -a "$WORK/espeak-ng/build/src/libespeak-ng/libespeak-ng.so"* "$OUT/"

echo
printf '%-30s %10s\n' FILE BYTES
find "$OUT" -type f -printf '%-30f %10s\n' | sort
TOTAL=$(find "$OUT" -type f -printf '%s\n' | paste -sd+ | bc)   # -type f skips the symlinks
awk -v t="$TOTAL" 'BEGIN { printf "%-30s %10d  (%.2f MB)\n", "TOTAL", t, t/1048576 }'

cat > "$OUT/README.txt" <<'EOF'
English-only espeak-ng, for phonemising English text on a device.

Point ESPEAK_DATA_PATH at espeak-ng-data/, then call espeak with IPA output — the API
equivalent of `espeak-ng -q --ipa -v en-us`. Verify that once against mobile/test_vectors.json:
the C API and the CLI are different call paths, and the model needs exactly the CLI's output.

Then tokenise the IPA against the model's symbol set: greedy longest-first, skip whitespace,
keep punctuation. See mobile/README.md.

espeak-ng is GPL-3.0.
EOF
echo
echo "bundle ready: $OUT"
