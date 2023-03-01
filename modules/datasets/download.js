// Copyright (c) 2022-2023, NVIDIA CORPORATION.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const fs = require('fs');
const Path = require('path');
const https = require('https');
const {spawn} = require('child_process');
const { finished } = require('stream/promises');
const { pipeline, PassThrough } = require('stream');

(async () => {

  const data = Path.join(__dirname, 'data');

  await fs.promises.access(data, fs.constants.F_OK).catch(() =>
  /**/  fs.promises.mkdir(data, { recursive: true, mode: `0755` }));

  const node_rapdids_data_s3 = `node-rapids-data.s3.us-west-2.amazonaws.com`;

  await checkOrDownload(
    { md5Hash: '301ab1cf44cacf3665f915738fdbb515', file: Path.join(data, 'graph.json') },
    { hostname: node_rapdids_data_s3, path: `/graph/graphology.json.gz` },
  );

  await checkOrDownload(
    { md5Hash: 'b63ef3847bc11d9568658ed77b95437e', file: Path.join(data, 'polys.arrow') },
    { hostname: node_rapdids_data_s3, path: `/spatial/263_tracts.arrow.gz` },
  );

  await checkOrDownload(
    { md5Hash: 'f67a641c60924f6828d1992e1a7fc46e', file: Path.join(data, 'points.arrow') },
    { hostname: node_rapdids_data_s3, path: `/spatial/168898952_points.arrow.gz` },
  );

  await checkOrDownload(
    { md5Hash: '268e7a8e7e4811f6bda8214981e9841c', file: Path.join(data, 'centerline.csv') },
    { hostname: node_rapdids_data_s3, path: `/spatial/nyc-centerline-02-2023.csv.gz` },
  );

})().catch((e) => console.error(e) || process.exit(1));

function checkOrDownload(
  { md5Hash, file },
  { hostname, path },
) {
  return fs.promises
    .stat(file, fs.constants.F_OK)
    .then(() => new Promise((resolve, reject) => {
      if (!md5Hash) { return resolve(); }
      const proc = spawn(`md5sum`, ["-c", "-"]);
      proc.stdin.write(`${md5Hash}  ${file}`);
      proc.stdin.end();
      proc.once('error', (err) => reject(err));
      proc.once('exit', (code, signal) => {
        (code || signal) ? reject(code || signal) : resolve();
      });
    }))
    .catch(() => new Promise((resolve, reject) =>
      finished(
        pipeline(
          download(hostname, path),
          fs.createWriteStream(file),
          (err) => err && reject(err)
        )
      ).then(resolve)
    ));
}

function download(hostname, path) {
  const options = {
    path,
    hostname,
    method: `GET`,
    headers: {
      [`Accept`]: `application/octet-stream`,
      [`Accept-Encoding`]: `br;q=1.0, gzip;q=0.8, deflate;q=0.6, identity;q=0.4, *;q=0.1`,
    },
  };

  const out = new PassThrough();

  https.request(options, (res) => {
    const encoding = res.headers['content-encoding'] || '';

    if (encoding.includes('gzip')) {
      res = res.pipe(require('zlib').createGunzip());
    } else if (encoding.includes('deflate')) {
      res = res.pipe(require('zlib').createInflate());
    } else if (encoding.includes('br')) {
      res = res.pipe(require('zlib').createBrotliDecompress());
    }

    pipeline(res, out, (err) => {
      if (err) {
        console.error(`Error loading "https://${[options.hostname, options.path].join('')}"\n${err}`);
      } else {
        console.log(`Loaded "https://${[options.hostname, options.path].join('')}"`);
      }
    });
  }).end();

  return out;
}
