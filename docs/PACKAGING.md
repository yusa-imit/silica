# Silica — Packaging Guide

This guide covers building and distributing Silica packages for various platforms.

## Overview

Silica provides native packages for:
- **Debian/Ubuntu** (.deb packages)
- **RHEL/Fedora/CentOS** (.rpm packages)
- **macOS** (Homebrew formula)

All packages include:
- Binary executable (`silica`)
- Configuration file (`silica.conf`)
- Systemd service file (Linux)
- Documentation (README, API, Getting Started, SQL Reference, Operations Guide, Architecture Guide)

---

## Prerequisites

### All Platforms
- **Zig 0.15.0+**: Required for building from source
- **Git**: For cloning the repository

### Debian/Ubuntu
```bash
sudo apt-get install debhelper devscripts zig
```

### RHEL/Fedora/CentOS
```bash
sudo dnf install rpm-build rpmdevtools zig
```

### macOS
```bash
brew install zig
```

---

## Building Packages

### Debian Package (.deb)

#### 1. Prepare Source Tarball
```bash
cd /path/to/silica
git archive --format=tar.gz --prefix=silica-0.3.0/ -o ../silica_0.3.0.orig.tar.gz HEAD
cd ..
tar -xzf silica_0.3.0.orig.tar.gz
cd silica-0.3.0
```

#### 2. Build Package
```bash
dpkg-buildpackage -us -uc
```

This will:
- Run `zig build -Doptimize=ReleaseSafe`
- Run `zig build test` (test suite)
- Install binary to `/usr/bin/silica`
- Install config to `/etc/silica/silica.conf`
- Install systemd service to `/lib/systemd/system/silica.service`
- Install docs to `/usr/share/doc/silica/`

#### 3. Output
```
../silica_0.3.0-1_amd64.deb
../silica_0.3.0-1_amd64.changes
../silica_0.3.0-1.debian.tar.xz
```

#### 4. Install Package
```bash
sudo dpkg -i ../silica_0.3.0-1_amd64.deb
```

#### 5. Verify Installation
```bash
silica --version
systemctl status silica  # Service status
cat /etc/silica/silica.conf  # Config file
```

---

### RPM Package (.rpm)

#### 1. Setup RPM Build Environment
```bash
rpmdev-setuptree
```

This creates:
```
~/rpmbuild/
  ├── BUILD/
  ├── RPMS/
  ├── SOURCES/
  ├── SPECS/
  └── SRPMS/
```

#### 2. Prepare Source Tarball
```bash
cd /path/to/silica
git archive --format=tar.gz --prefix=silica-0.3.0/ -o ~/rpmbuild/SOURCES/silica-0.3.0.tar.gz HEAD
```

#### 3. Copy Spec File
```bash
cp packaging/rpm/silica.spec ~/rpmbuild/SPECS/
```

#### 4. Build Package
```bash
cd ~/rpmbuild/SPECS
rpmbuild -ba silica.spec
```

This will:
- Extract source to `~/rpmbuild/BUILD/silica-0.3.0/`
- Run `zig build -Doptimize=ReleaseSafe`
- Run `zig build test` (test suite)
- Install binary to `/usr/bin/silica`
- Install config to `/etc/silica/silica.conf`
- Install systemd service to `/usr/lib/systemd/system/silica.service`
- Install docs to `/usr/share/doc/silica/`
- Create `silica` user and group

#### 5. Output
```
~/rpmbuild/RPMS/x86_64/silica-0.3.0-1.el8.x86_64.rpm
~/rpmbuild/SRPMS/silica-0.3.0-1.el8.src.rpm
```

#### 6. Install Package
```bash
sudo dnf install ~/rpmbuild/RPMS/x86_64/silica-0.3.0-1.el8.x86_64.rpm
```

#### 7. Verify Installation
```bash
silica --version
systemctl status silica
cat /etc/silica/silica.conf
```

---

### Homebrew Formula (macOS)

#### 1. Publish Release
First, create a GitHub release (this is required for Homebrew):
```bash
git tag -a v0.3.0 -m "Release v0.3.0: Phase 3 complete"
git push origin v0.3.0
gh release create v0.3.0 --title "v0.3.0: WAL & Basic Transactions" --notes "Release notes..."
```

#### 2. Calculate SHA256
```bash
curl -L https://github.com/yusa-imit/silica/archive/refs/tags/v0.3.0.tar.gz | shasum -a 256
```

Update `packaging/homebrew/silica.rb` with the actual SHA256.

#### 3. Submit to Homebrew (Official)
To publish to homebrew-core:
```bash
# Fork homebrew/homebrew-core
git clone https://github.com/Homebrew/homebrew-core.git
cd homebrew-core
git checkout -b silica-0.3.0

# Copy formula
cp /path/to/silica/packaging/homebrew/silica.rb Formula/silica.rb

# Commit and create PR
git add Formula/silica.rb
git commit -m "silica 0.3.0 (new formula)"
git push origin silica-0.3.0
# Create PR on GitHub
```

#### 4. Or: Install from Local Formula (Testing)
```bash
brew install --build-from-source packaging/homebrew/silica.rb
```

#### 5. Verify Installation
```bash
silica --version
brew services start silica  # Start server
brew services list  # Check status
```

---

## Package Contents

### Installed Files

| Path | Description |
|------|-------------|
| `/usr/bin/silica` | Main executable |
| `/etc/silica/silica.conf` | Configuration file |
| `/lib/systemd/system/silica.service` | Systemd service (Linux) |
| `/usr/share/doc/silica/README.md` | Project overview |
| `/usr/share/doc/silica/API_REFERENCE.md` | Zig API documentation |
| `/usr/share/doc/silica/GETTING_STARTED.md` | Tutorial |
| `/usr/share/doc/silica/SQL_REFERENCE.md` | SQL syntax guide |
| `/usr/share/doc/silica/OPERATIONS_GUIDE.md` | Operations manual |
| `/usr/share/doc/silica/ARCHITECTURE_GUIDE.md` | Internal design docs |
| `/var/lib/silica/` | Data directory (created by package) |

### Systemd Service

The service runs as user `silica` with:
- Data directory: `/var/lib/silica`
- Port: `5433` (default)
- Restart policy: `on-failure` (auto-restart on crashes)
- Security hardening: `NoNewPrivileges`, `PrivateTmp`, `ProtectSystem=strict`

Start the service:
```bash
sudo systemctl enable silica
sudo systemctl start silica
```

Check status:
```bash
sudo systemctl status silica
journalctl -u silica -f  # View logs
```

---

## Updating Packages for New Releases

### 1. Update Version Numbers

**Debian (`debian/changelog`)**:
```bash
dch --newversion 0.4.0-1 "Release 0.4.0: New features..."
```

**RPM (`packaging/rpm/silica.spec`)**:
```spec
Version:        0.4.0
Release:        1%{?dist}

%changelog
* Fri Apr 05 2026 Yusa <yusa@example.com> - 0.4.0-1
- Release 0.4.0: MVCC and advanced transactions
- New features: ...
```

**Homebrew (`packaging/homebrew/silica.rb`)**:
```ruby
url "https://github.com/yusa-imit/silica/archive/refs/tags/v0.4.0.tar.gz"
sha256 "NEW_SHA256_HERE"
```

### 2. Rebuild Packages
Follow the build steps above for each platform.

### 3. Test Packages
- Install on clean VM/container
- Run test suite: `silica --version`
- Start server: `systemctl start silica` (Linux) or `brew services start silica` (macOS)
- Run smoke tests: create database, insert data, query

---

## Distribution

### APT Repository (Debian/Ubuntu)

#### 1. Setup Repository Structure
```bash
mkdir -p repo/pool/main/s/silica
mkdir -p repo/dists/stable/main/binary-amd64
```

#### 2. Copy .deb File
```bash
cp silica_0.3.0-1_amd64.deb repo/pool/main/s/silica/
```

#### 3. Generate Packages Index
```bash
cd repo
dpkg-scanpackages pool/ /dev/null | gzip -9c > dists/stable/main/binary-amd64/Packages.gz
```

#### 4. Generate Release File
```bash
cd dists/stable
apt-ftparchive release . > Release
gpg --clearsign -o InRelease Release  # Optional: sign with GPG
```

#### 5. Serve via HTTP
```bash
# Option 1: GitHub Pages
git init
git add .
git commit -m "Initial APT repository"
git branch -M gh-pages
git remote add origin https://github.com/yusa-imit/silica-apt.git
git push -u origin gh-pages

# Option 2: S3/CDN
aws s3 sync repo/ s3://silica-apt-repo/ --acl public-read
```

#### 6. Users Add Repository
```bash
echo "deb https://yusa-imit.github.io/silica-apt stable main" | sudo tee /etc/apt/sources.list.d/silica.list
sudo apt-get update
sudo apt-get install silica
```

---

### YUM Repository (RHEL/Fedora/CentOS)

#### 1. Setup Repository Structure
```bash
mkdir -p repo/el8/x86_64/Packages
```

#### 2. Copy .rpm File
```bash
cp ~/rpmbuild/RPMS/x86_64/silica-0.3.0-1.el8.x86_64.rpm repo/el8/x86_64/Packages/
```

#### 3. Generate Repository Metadata
```bash
cd repo/el8/x86_64
createrepo .
```

#### 4. Serve via HTTP
```bash
# Option 1: GitHub Pages
git init
git add .
git commit -m "Initial YUM repository"
git branch -M gh-pages
git remote add origin https://github.com/yusa-imit/silica-yum.git
git push -u origin gh-pages

# Option 2: S3/CDN
aws s3 sync repo/ s3://silica-yum-repo/ --acl public-read
```

#### 5. Users Add Repository
```bash
sudo tee /etc/yum.repos.d/silica.repo <<EOF
[silica]
name=Silica Database Repository
baseurl=https://yusa-imit.github.io/silica-yum/el8/x86_64
enabled=1
gpgcheck=0
EOF

sudo dnf install silica
```

---

### Homebrew Tap (macOS)

#### 1. Create Tap Repository
```bash
mkdir homebrew-silica
cd homebrew-silica
mkdir Formula
cp ../silica/packaging/homebrew/silica.rb Formula/
git init
git add .
git commit -m "Initial tap: silica formula"
git remote add origin https://github.com/yusa-imit/homebrew-silica.git
git push -u origin main
```

#### 2. Users Install from Tap
```bash
brew tap yusa-imit/silica
brew install silica
```

Or directly:
```bash
brew install yusa-imit/silica/silica
```

---

## CI/CD Integration

### GitHub Actions Workflow

Add to `.github/workflows/release.yml`:

```yaml
name: Release Packages

on:
  release:
    types: [published]

jobs:
  build-deb:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y debhelper devscripts
          # Install Zig
          wget https://ziglang.org/download/0.15.2/zig-linux-x86_64-0.15.2.tar.xz
          tar -xf zig-linux-x86_64-0.15.2.tar.xz
          echo "$PWD/zig-linux-x86_64-0.15.2" >> $GITHUB_PATH
      - name: Build .deb package
        run: |
          dpkg-buildpackage -us -uc
          mv ../silica_*.deb .
      - name: Upload to release
        uses: actions/upload-release-asset@v1
        with:
          upload_url: ${{ github.event.release.upload_url }}
          asset_path: silica_${{ github.event.release.tag_name }}_amd64.deb
          asset_name: silica_${{ github.event.release.tag_name }}_amd64.deb
          asset_content_type: application/vnd.debian.binary-package

  build-rpm:
    runs-on: ubuntu-latest
    container: fedora:latest
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: |
          dnf install -y rpm-build rpmdevtools wget tar xz
          # Install Zig
          wget https://ziglang.org/download/0.15.2/zig-linux-x86_64-0.15.2.tar.xz
          tar -xf zig-linux-x86_64-0.15.2.tar.xz
          echo "$PWD/zig-linux-x86_64-0.15.2" >> $GITHUB_PATH
      - name: Setup RPM build tree
        run: rpmdev-setuptree
      - name: Build .rpm package
        run: |
          VERSION="${GITHUB_REF#refs/tags/v}"
          git archive --format=tar.gz --prefix=silica-$VERSION/ -o ~/rpmbuild/SOURCES/silica-$VERSION.tar.gz HEAD
          cp packaging/rpm/silica.spec ~/rpmbuild/SPECS/
          rpmbuild -ba ~/rpmbuild/SPECS/silica.spec
          cp ~/rpmbuild/RPMS/x86_64/silica-*.rpm .
      - name: Upload to release
        uses: actions/upload-release-asset@v1
        with:
          upload_url: ${{ github.event.release.upload_url }}
          asset_path: silica-${{ github.event.release.tag_name }}-1.fc*.x86_64.rpm
          asset_name: silica-${{ github.event.release.tag_name }}-1.fc.x86_64.rpm
          asset_content_type: application/x-rpm

  homebrew-tap:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
        with:
          repository: yusa-imit/homebrew-silica
      - name: Update formula
        run: |
          VERSION="${GITHUB_REF#refs/tags/v}"
          SHA256=$(curl -L https://github.com/yusa-imit/silica/archive/refs/tags/v$VERSION.tar.gz | shasum -a 256 | cut -d' ' -f1)
          sed -i '' "s|url \".*\"|url \"https://github.com/yusa-imit/silica/archive/refs/tags/v$VERSION.tar.gz\"|" Formula/silica.rb
          sed -i '' "s|sha256 \".*\"|sha256 \"$SHA256\"|" Formula/silica.rb
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add Formula/silica.rb
          git commit -m "silica $VERSION"
          git push
```

---

## Troubleshooting

### Build Failures

**Debian: `dh_auto_build` fails**
- Check Zig installation: `zig version`
- Verify dependencies: `dpkg-checkbuilddeps`
- Clean build: `rm -rf zig-out .zig-cache && zig build`

**RPM: `rpmbuild` fails**
- Check spec file syntax: `rpmlint ~/rpmbuild/SPECS/silica.spec`
- Verify source tarball: `tar -tzf ~/rpmbuild/SOURCES/silica-0.3.0.tar.gz | head`
- Check build log: `less ~/rpmbuild/BUILD/silica-0.3.0/build.log`

**Homebrew: Formula audit fails**
- Run audit: `brew audit --strict --online packaging/homebrew/silica.rb`
- Test installation: `brew install --build-from-source packaging/homebrew/silica.rb`
- Check logs: `brew gist-logs silica`

### Installation Failures

**Debian: Dependencies not met**
```bash
sudo apt-get install -f  # Fix broken dependencies
```

**RPM: Conflicts with existing package**
```bash
sudo dnf remove old-package
sudo dnf install silica
```

**Homebrew: Zig not found**
```bash
brew install zig
brew reinstall silica
```

### Service Failures

**Systemd: Service won't start**
```bash
journalctl -u silica -n 50  # Check logs
systemctl cat silica  # View service file
sudo -u silica /usr/bin/silica server --data-dir /var/lib/silica --port 5433  # Test manually
```

**Permissions: Data directory not writable**
```bash
sudo chown -R silica:silica /var/lib/silica
sudo chmod 0755 /var/lib/silica
```

---

## Package Signing

### GPG Signing (Debian)

#### 1. Generate GPG Key (if needed)
```bash
gpg --full-generate-key
# Choose RSA, 4096 bits, no expiration
# Use maintainer email: yusa@example.com
```

#### 2. Sign Package
```bash
dpkg-buildpackage -k<KEY_ID>
```

#### 3. Export Public Key
```bash
gpg --armor --export <KEY_ID> > silica-archive-keyring.gpg
```

#### 4. Users Import Key
```bash
sudo apt-key add silica-archive-keyring.gpg
```

---

### RPM Signing

#### 1. Generate GPG Key (same as Debian)
```bash
gpg --full-generate-key
```

#### 2. Import Key to RPM
```bash
gpg --export -a '<KEY_ID>' > RPM-GPG-KEY-silica
sudo rpm --import RPM-GPG-KEY-silica
```

#### 3. Sign RPM
```bash
echo "%_gpg_name <KEY_ID>" >> ~/.rpmmacros
rpmsign --addsign ~/rpmbuild/RPMS/x86_64/silica-0.3.0-1.el8.x86_64.rpm
```

#### 4. Verify Signature
```bash
rpm --checksig ~/rpmbuild/RPMS/x86_64/silica-0.3.0-1.el8.x86_64.rpm
```

---

## References

- [Debian New Maintainers' Guide](https://www.debian.org/doc/manuals/maint-guide/)
- [Fedora Packaging Guidelines](https://docs.fedoraproject.org/en-US/packaging-guidelines/)
- [Homebrew Formula Cookbook](https://docs.brew.sh/Formula-Cookbook)
- [Zig Build System](https://ziglang.org/learn/build-system/)

---

## Support

For packaging issues:
- GitHub Issues: https://github.com/yusa-imit/silica/issues
- Mailing list: silica-dev@example.com (coming soon)
- IRC: #silica on libera.chat (coming soon)
