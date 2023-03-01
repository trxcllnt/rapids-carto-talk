# Usage

```shell
# Install workspace dependencies
yarn

# Download datasets and compile TS -> JS
yarn workspaces run build

# Run cuSpatial point-in-polygon demo
yarn workspace @rapids-carto-talk/nyc-taxi run pip

# Run cuDF + Carto NYC streets centerline demo
yarn workspace @rapids-carto-talk/nyc-taxi run streets

# Run cuDF + cuGraph parse/layout/visualization demo
yarn workspace @rapids-carto-talk/json-graph run graph
```
