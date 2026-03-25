Name:           silica
Version:        0.3.0
Release:        1%{?dist}
Summary:        Production-grade embedded relational database

License:        MIT
URL:            https://github.com/yusa-imit/silica
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  zig >= 0.15.0
Requires:       glibc

%description
Silica is a full-featured RDBMS written in Zig, inspired by SQLite and
PostgreSQL. It supports dual-mode operation (embedded + client-server),
MVCC transactions, full SQL:2016 compliance, streaming replication, and
PostgreSQL wire protocol compatibility.

Features:
- ACID transactions with MVCC (READ COMMITTED, REPEATABLE READ, SERIALIZABLE)
- Full SQL support (DDL, DML, DQL, CTEs, window functions, JSON, full-text search)
- Multiple index types (B+Tree, Hash, GIN, GiST)
- Write-Ahead Logging (WAL) with crash recovery
- Streaming replication with hot standby
- PostgreSQL wire protocol v3 for client compatibility
- Embedded mode (single library, no server process)
- Server mode (multi-client, TCP-based)

%prep
%setup -q

%build
zig build -Doptimize=ReleaseSafe

%install
rm -rf %{buildroot}

# Install binary
install -D -m 0755 zig-out/bin/silica %{buildroot}%{_bindir}/silica

# Install configuration
install -D -m 0644 silica.conf.example %{buildroot}%{_sysconfdir}/silica/silica.conf

# Install systemd service
install -D -m 0644 packaging/systemd/silica.service %{buildroot}%{_unitdir}/silica.service

# Install documentation
install -D -m 0644 README.md %{buildroot}%{_docdir}/%{name}/README.md
install -D -m 0644 docs/API_REFERENCE.md %{buildroot}%{_docdir}/%{name}/API_REFERENCE.md
install -D -m 0644 docs/GETTING_STARTED.md %{buildroot}%{_docdir}/%{name}/GETTING_STARTED.md
install -D -m 0644 docs/SQL_REFERENCE.md %{buildroot}%{_docdir}/%{name}/SQL_REFERENCE.md
install -D -m 0644 docs/OPERATIONS_GUIDE.md %{buildroot}%{_docdir}/%{name}/OPERATIONS_GUIDE.md
install -D -m 0644 docs/ARCHITECTURE_GUIDE.md %{buildroot}%{_docdir}/%{name}/ARCHITECTURE_GUIDE.md

# Create data directory
install -d -m 0755 %{buildroot}%{_sharedstatedir}/silica

%check
zig build test

%pre
# Create silica user and group
getent group silica >/dev/null || groupadd -r silica
getent passwd silica >/dev/null || \
    useradd -r -g silica -d %{_sharedstatedir}/silica -s /sbin/nologin \
    -c "Silica Database Server" silica
exit 0

%post
%systemd_post silica.service

%preun
%systemd_preun silica.service

%postun
%systemd_postun_with_restart silica.service

%files
%license debian/copyright
%doc README.md
%{_bindir}/silica
%config(noreplace) %{_sysconfdir}/silica/silica.conf
%{_unitdir}/silica.service
%{_docdir}/%{name}/
%attr(0755,silica,silica) %dir %{_sharedstatedir}/silica

%changelog
* Mon Mar 25 2026 Yusa <yusa@example.com> - 0.3.0-1
- Initial RPM package release
- Phase 3: WAL & Basic Transactions complete
- Full ACID compliance with WAL-based durability
- Crash recovery and checkpoint support
- B+Tree storage engine with buffer pool
- SQL frontend with tokenizer, parser, planner, optimizer, executor
- Transaction support (BEGIN, COMMIT, ROLLBACK)
