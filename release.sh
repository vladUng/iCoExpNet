# Bump version in pyproject.toml
git commit -am "release: bump to v0.1.12"

# Tag it
git tag v0.1.12

# Push code + tag
git push origin main
git push origin v0.1.12