// Copyright (c) 2022, NVIDIA CORPORATION.
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
const { finished } = require('stream/promises');
const { pipeline, PassThrough } = require('stream');

(async () => {

  await fs.promises.access(Path.join(__dirname, 'data'), fs.constants.F_OK).catch(() =>
  /**/  fs.promises.mkdir(Path.join(__dirname, 'data'), { recursive: true, mode: `0755` }));

  const node_rapdids_data_s3 = `node-rapids-data.s3.us-west-2.amazonaws.com`;

  await finished(pipeline(
    download(node_rapdids_data_s3, `/graph/graphology.json.gz`),
    fs.createWriteStream(Path.join(__dirname, 'data', 'graph.json')),
    (err) => { }
  ));

  await finished(pipeline(
    download(node_rapdids_data_s3, `/spatial/263_tracts.arrow.gz`),
    fs.createWriteStream(Path.join(__dirname, 'data', 'polys.arrow')),
    (err) => { }
  ));

  await finished(pipeline(
    download(node_rapdids_data_s3, `/spatial/168898952_points.arrow.gz`),
    fs.createWriteStream(Path.join(__dirname, 'data', 'points.arrow')),
    (err) => { }
  ));

  await finished(pipeline(
    download(`data.cityofnewyork.us`, `/api/views/8rma-cm9c/rows.csv?accessType=DOWNLOAD`),
    fs.createWriteStream(Path.join(__dirname, 'data', 'centerline.csv')),
    (err) => { }
  ));

})().catch((e) => console.error(e) || process.exit(1));

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
