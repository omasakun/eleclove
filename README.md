<!-- TODO: Translation -->
<div align="center">
  <h1>Eleclove</h1>
  <p>電子回路の単純なシミュレーター</p>
</div>
<br>
<br>

<!-- TODO: Add screenshots here -->

## 使い方

TODO

## 開発のはじめかた

Linux や WSL2 での開発を想定しています。

```bash
# インストールしていない場合は asdf と pnpm を入れておきます。
# asdf: https://asdf-vm.com/guide/getting-started.html
# pnpm: https://pnpm.io/installation

# 必要なツールなどをインストールします。
asdf install
pdm  install -G :all
pnpm install

# これで準備完了です！
source .venv/bin/activate
python -m eleclove
```
