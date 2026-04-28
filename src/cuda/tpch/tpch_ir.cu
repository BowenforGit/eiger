#include "tpch/tpch_ir.hpp"

#include "ir/expr.hpp"

namespace gamax::ir::tpch {

namespace {

// ── Short aliases, kept local to this file ──
using gamax::data_type;
using gamax::type_id;
data_type t(type_id id) { return data_type{id}; }
const data_type t_key    = t(type_id::INT64);
const data_type t_int    = t(type_id::INT32);
const data_type t_date   = t(type_id::INT32);
const data_type t_money  = t(type_id::FLOAT64);
const data_type t_str    = t(type_id::STRING);

// Convenience: year(date) — TPC-H Q7/Q8/Q9 need `extract(year from date)`.
// Kept here as a file-local builder rather than baked into the IR
// proper, because the function-call surface (ir::func) is intentionally
// generic. A physical lowering pass will rewrite "year" into whatever
// arithmetic the backend supports.
Expr year_of(Expr date_expr) {
    return func("year", {std::move(date_expr)}, t_int);
}

// Convenience: register one TPC-H table in the catalog. `cols` is a
// list of {name, type}; all columns are marked non-nullable (TPC-H).
void add_table(
    Catalog& cat,
    std::string name,
    std::vector<std::pair<std::string, data_type>> cols)
{
    CatalogTable ct;
    ct.name = std::move(name);
    for (auto& [cn, ct_ty] : cols) {
        ct.column_names.push_back(std::move(cn));
        ct.column_types.push_back(ct_ty);
        ct.column_nullable.push_back(false);
    }
    cat.add_table(std::move(ct));
}

} // namespace

Catalog make_tpch_catalog() {
    Catalog cat;

    // region (r_regionkey, r_name, r_comment)
    add_table(cat, "region", {
        {"r_regionkey", t_key},
        {"r_name",      t_str},
        {"r_comment",   t_str},
    });

    // nation (n_nationkey, n_name, n_regionkey, n_comment)
    add_table(cat, "nation", {
        {"n_nationkey", t_key},
        {"n_name",      t_str},
        {"n_regionkey", t_key},
        {"n_comment",   t_str},
    });

    // part (p_partkey, p_name, p_mfgr, p_brand, p_type, p_size,
    //       p_container, p_retailprice, p_comment)
    add_table(cat, "part", {
        {"p_partkey",     t_key},
        {"p_name",        t_str},
        {"p_mfgr",        t_str},
        {"p_brand",       t_str},
        {"p_type",        t_str},
        {"p_size",        t_int},
        {"p_container",   t_str},
        {"p_retailprice", t_money},
        {"p_comment",     t_str},
    });

    // supplier (s_suppkey, s_name, s_address, s_nationkey, s_phone,
    //           s_acctbal, s_comment)
    add_table(cat, "supplier", {
        {"s_suppkey",   t_key},
        {"s_name",      t_str},
        {"s_address",   t_str},
        {"s_nationkey", t_key},
        {"s_phone",     t_str},
        {"s_acctbal",   t_money},
        {"s_comment",   t_str},
    });

    // partsupp (ps_partkey, ps_suppkey, ps_availqty, ps_supplycost,
    //           ps_comment)
    add_table(cat, "partsupp", {
        {"ps_partkey",    t_key},
        {"ps_suppkey",    t_key},
        {"ps_availqty",   t_int},
        {"ps_supplycost", t_money},
        {"ps_comment",    t_str},
    });

    // customer (c_custkey, c_name, c_address, c_nationkey, c_phone,
    //           c_acctbal, c_mktsegment, c_comment)
    add_table(cat, "customer", {
        {"c_custkey",    t_key},
        {"c_name",       t_str},
        {"c_address",    t_str},
        {"c_nationkey",  t_key},
        {"c_phone",      t_str},
        {"c_acctbal",    t_money},
        {"c_mktsegment", t_str},
        {"c_comment",    t_str},
    });

    // orders (o_orderkey, o_custkey, o_orderstatus, o_totalprice,
    //         o_orderdate, o_orderpriority, o_clerk, o_shippriority,
    //         o_comment)
    add_table(cat, "orders", {
        {"o_orderkey",      t_key},
        {"o_custkey",       t_key},
        {"o_orderstatus",   t_str},
        {"o_totalprice",    t_money},
        {"o_orderdate",     t_date},
        {"o_orderpriority", t_str},
        {"o_clerk",         t_str},
        {"o_shippriority",  t_int},
        {"o_comment",       t_str},
    });

    // lineitem (l_orderkey, l_partkey, l_suppkey, l_linenumber,
    //           l_quantity, l_extendedprice, l_discount, l_tax,
    //           l_returnflag, l_linestatus, l_shipdate, l_commitdate,
    //           l_receiptdate, l_shipinstruct, l_shipmode, l_comment)
    add_table(cat, "lineitem", {
        {"l_orderkey",      t_key},
        {"l_partkey",       t_key},
        {"l_suppkey",       t_key},
        {"l_linenumber",    t_int},
        {"l_quantity",      t_money},
        {"l_extendedprice", t_money},
        {"l_discount",      t_money},
        {"l_tax",           t_money},
        {"l_returnflag",    t_str},
        {"l_linestatus",    t_str},
        {"l_shipdate",      t_date},
        {"l_commitdate",    t_date},
        {"l_receiptdate",   t_date},
        {"l_shipinstruct",  t_str},
        {"l_shipmode",      t_str},
        {"l_comment",       t_str},
    });

    return cat;
}

// ── Q1: Pricing Summary Report ─────────────────────────────────────
//
//   SELECT l_returnflag, l_linestatus,
//          sum(l_quantity), sum(l_extendedprice),
//          sum(l_extendedprice*(1-l_discount)),
//          sum(l_extendedprice*(1-l_discount)*(1+l_tax)),
//          avg(l_quantity), avg(l_extendedprice), avg(l_discount),
//          count(*)
//   FROM lineitem
//   WHERE l_shipdate <= date '[date]'
//   GROUP BY l_returnflag, l_linestatus
//   ORDER BY l_returnflag, l_linestatus;
PlanPtr build_q1(const QueryBuilder& qb) {
    auto li = qb.scan("lineitem", {
        "l_shipdate", "l_returnflag", "l_linestatus",
        "l_quantity", "l_extendedprice", "l_discount", "l_tax",
    });

    // Filter on l_shipdate first — at SF=100 this drops the project's
    // input from 600 M to ~570 M rows, but more importantly it consumes
    // l_shipdate immediately so the project never has to carry an
    // extra 2.4 GiB int32 column. Then project derived disc_price /
    // charge so that the subsequent aggregate's bodies become pure
    // ColRef references (what the emitter expects).
    auto filtered = li.filter(
        li.c("l_shipdate") <= lit(19980902));  // arbitrary cutoff
    auto projected = filtered.project({
        {"l_returnflag",     filtered.c("l_returnflag")},
        {"l_linestatus",     filtered.c("l_linestatus")},
        {"l_quantity",       filtered.c("l_quantity")},
        {"l_extendedprice",  filtered.c("l_extendedprice")},
        {"l_discount",       filtered.c("l_discount")},
        {"disc_price",
            filtered.c("l_extendedprice") *
            (lit(1.0) - filtered.c("l_discount"))},
        {"charge",
            filtered.c("l_extendedprice") *
            (lit(1.0) - filtered.c("l_discount")) *
            (lit(1.0) + filtered.c("l_tax"))},
    });

    // sum_qty / sum_base_price already cover l_quantity and
    // l_extendedprice; running avg(l_quantity) and avg(l_extendedprice)
    // alongside their sums forces the pre-groupby project to
    // materialize each of those columns twice — at SF=100 that
    // duplication blows ~8.8 GiB of intermediates and OOMs the slab.
    // Keep one materialization per source column here, and recover
    // avg_qty / avg_price downstream from sum/count.
    auto agg = projected.aggregate(
        {"l_returnflag", "l_linestatus"},
        {
            {"sum_qty",        AggKind::Sum, projected.c("l_quantity")},
            {"sum_base_price", AggKind::Sum, projected.c("l_extendedprice")},
            {"sum_disc_price", AggKind::Sum, projected.c("disc_price")},
            {"sum_charge",     AggKind::Sum, projected.c("charge")},
            {"avg_disc",       AggKind::Avg, projected.c("l_discount")},
            {"count_order",    AggKind::CountAll, lit(int64_t{0})},
        });

    // Compute avg_qty / avg_price after the aggregate. count_order is
    // int64; the division promotes to double automatically.
    auto post = agg.project({
        {"l_returnflag",   agg.c("l_returnflag")},
        {"l_linestatus",   agg.c("l_linestatus")},
        {"sum_qty",        agg.c("sum_qty")},
        {"sum_base_price", agg.c("sum_base_price")},
        {"sum_disc_price", agg.c("sum_disc_price")},
        {"sum_charge",     agg.c("sum_charge")},
        {"avg_qty",        agg.c("sum_qty")        / agg.c("count_order")},
        {"avg_price",      agg.c("sum_base_price") / agg.c("count_order")},
        {"avg_disc",       agg.c("avg_disc")},
        {"count_order",    agg.c("count_order")},
    });

    auto sorted = post.sort({asc("l_returnflag"), asc("l_linestatus")});
    return sorted.plan();
}

// ── Q2: Minimum Cost Supplier ──────────────────────────────────────
//
// Five-way join (part, supplier, partsupp, nation, region) with size,
// p_type endswith '[type]', and r_name filters. The correlated subquery
// "ps_supplycost = MIN(ps_supplycost) within partsupp+supplier+nation+
//  region filtered to the region" is lowered to a second plan that
// computes the per-partkey min, joined back to the outer plan on
// (ps_partkey, min_supplycost). That's a multi-key join — our IR
// represents it (the emitter currently throws on multi-key).
//
// ORDER BY s_acctbal DESC, n_name, s_name, p_partkey  LIMIT 100.
PlanPtr build_q2(const QueryBuilder& qb) {
    auto p = qb.scan("part",
        {"p_partkey", "p_mfgr", "p_type", "p_size"});
    auto s = qb.scan("supplier",
        {"s_suppkey", "s_name", "s_address", "s_nationkey",
         "s_phone", "s_acctbal", "s_comment"});
    auto ps = qb.scan("partsupp",
        {"ps_partkey", "ps_suppkey", "ps_supplycost"});
    auto n = qb.scan("nation", {"n_nationkey", "n_name", "n_regionkey"});
    auto r = qb.scan("region", {"r_regionkey", "r_name"});

    // Filter part size and type suffix.
    auto p_filt = p
        .filter(p.c("p_size") == lit(15))
        .filter(string_op(StringOpKind::Endswith,
                          p.c("p_type"), {"BRASS"}));
    // Filter region by name.
    auto r_filt = r.filter(
        string_op(StringOpKind::Equal, r.c("r_name"), {"EUROPE"}));

    // region ⋈ nation ⋈ supplier ⋈ partsupp ⋈ part.
    auto rn = r_filt.inner_join(n, r_filt.c("r_regionkey") == n.c("n_regionkey"));
    auto rns = rn.inner_join(s, rn.c("n_nationkey") == s.c("s_nationkey"));
    auto rnsp = rns.inner_join(ps, rns.c("s_suppkey") == ps.c("ps_suppkey"));
    auto joined = rnsp.inner_join(
        p_filt, rnsp.c("ps_partkey") == p_filt.c("p_partkey"));

    // Per-partkey min supplycost for the same region filter — built as
    // a sibling plan, joined back on (partkey, min_cost).
    auto ps2 = qb.scan("partsupp", {"ps_partkey", "ps_suppkey", "ps_supplycost"});
    auto s2  = qb.scan("supplier", {"s_suppkey", "s_nationkey"});
    auto n2  = qb.scan("nation",   {"n_nationkey", "n_regionkey"});
    auto r2  = qb.scan("region",   {"r_regionkey", "r_name"});
    auto r2_filt = r2.filter(
        string_op(StringOpKind::Equal, r2.c("r_name"), {"EUROPE"}));
    auto rn_inner  = r2_filt.inner_join(n2,
        r2_filt.c("r_regionkey") == n2.c("n_regionkey"));
    auto rns_inner = rn_inner.inner_join(s2,
        rn_inner.c("n_nationkey") == s2.c("s_nationkey"));
    auto rnsp_inner = rns_inner.inner_join(ps2,
        rns_inner.c("s_suppkey") == ps2.c("ps_suppkey"));
    auto min_costs = rnsp_inner.aggregate(
        {"ps_partkey"},
        {{"min_cost", AggKind::Min, rnsp_inner.c("ps_supplycost")}});

    // Outer ⋈ min_costs on (ps_partkey, ps_supplycost) == (ps_partkey, min_cost).
    // This is a multi-key join — IR represents it; emitter throws.
    auto final_join = joined.inner_join(min_costs,
          (joined.c("ps_partkey")    == min_costs.c("ps_partkey"))
       && (joined.c("ps_supplycost") == min_costs.c("min_cost")));

    auto projected = final_join.project({
        {"s_acctbal",  final_join.c("s_acctbal")},
        {"s_name",     final_join.c("s_name")},
        {"n_name",     final_join.c("n_name")},
        {"p_partkey",  final_join.c("p_partkey")},
        {"p_mfgr",     final_join.c("p_mfgr")},
        {"s_address",  final_join.c("s_address")},
        {"s_phone",    final_join.c("s_phone")},
        {"s_comment",  final_join.c("s_comment")},
    });

    auto sorted = projected.sort({
        desc("s_acctbal"), asc("n_name"), asc("s_name"), asc("p_partkey"),
    });
    return sorted.limit(100).plan();
}

// ── Q3: Shipping Priority ──────────────────────────────────────────
//
//   SELECT l_orderkey,
//          sum(l_extendedprice * (1 - l_discount)) as revenue,
//          o_orderdate, o_shippriority
//   FROM customer, orders, lineitem
//   WHERE c_mktsegment = 'BUILDING'
//     AND c_custkey = o_custkey
//     AND l_orderkey = o_orderkey
//     AND o_orderdate < date '[date]'
//     AND l_shipdate  > date '[date]'
//   GROUP BY l_orderkey, o_orderdate, o_shippriority
//   ORDER BY revenue DESC, o_orderdate  LIMIT 10;
PlanPtr build_q3(const QueryBuilder& qb) {
    auto c = qb.scan("customer",
        {"c_custkey", "c_mktsegment"});
    auto o = qb.scan("orders",
        {"o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"});
    auto l = qb.scan("lineitem",
        {"l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"});

    auto c_filt = c.filter(
        string_op(StringOpKind::Equal, c.c("c_mktsegment"), {"BUILDING"}));
    auto o_filt = o.filter(o.c("o_orderdate") < lit(19950315));
    auto l_filt = l.filter(l.c("l_shipdate")  > lit(19950315));

    auto co  = c_filt.inner_join(o_filt,
        c_filt.c("c_custkey") == o_filt.c("o_custkey"));
    auto col = co.inner_join(l_filt,
        l_filt.c("l_orderkey") == co.c("o_orderkey"));

    auto projected = col.project({
        {"l_orderkey",      col.c("l_orderkey")},
        {"o_orderdate",     col.c("o_orderdate")},
        {"o_shippriority",  col.c("o_shippriority")},
        {"disc_price",
            col.c("l_extendedprice") * (lit(1.0) - col.c("l_discount"))},
    });
    auto agg = projected.aggregate(
        {"l_orderkey", "o_orderdate", "o_shippriority"},
        {{"revenue", AggKind::Sum, projected.c("disc_price")}});
    // Spec's SELECT column order is (l_orderkey, revenue, o_orderdate,
    // o_shippriority); the aggregate naturally produces them in
    // key-first order, so a pure-ColRef Project here re-orders
    // without renumbering any ColumnIds — the subsequent Sort still
    // resolves `revenue` and `o_orderdate` by the same ids.
    auto out = agg.project({
        {"l_orderkey",     agg.c("l_orderkey")},
        {"revenue",        agg.c("revenue")},
        {"o_orderdate",    agg.c("o_orderdate")},
        {"o_shippriority", agg.c("o_shippriority")},
    });
    return out.sort({desc("revenue"), asc("o_orderdate")}).limit(10).plan();
}

// ── Q4: Order Priority Checking ────────────────────────────────────
//
//   SELECT o_orderpriority, count(*) as order_count
//   FROM orders
//   WHERE o_orderdate >= date '[d]'
//     AND o_orderdate <  date '[d]' + interval '3' month
//     AND EXISTS (SELECT * FROM lineitem
//                 WHERE l_orderkey = o_orderkey
//                   AND l_commitdate < l_receiptdate)
//   GROUP BY o_orderpriority
//   ORDER BY o_orderpriority;
//
// EXISTS is the classical semi-join lowering: filter lineitem first,
// then LEFT SEMI JOIN orders against the filtered lineitem on orderkey.
PlanPtr build_q4(const QueryBuilder& qb) {
    auto o = qb.scan("orders", {"o_orderkey", "o_orderdate", "o_orderpriority"});
    auto l = qb.scan("lineitem", {"l_orderkey", "l_commitdate", "l_receiptdate"});

    auto o_filt = o.filter(
            (o.c("o_orderdate") >= lit(19930701))
         && (o.c("o_orderdate") <  lit(19931001)));
    auto l_filt = l.filter(l.c("l_commitdate") < l.c("l_receiptdate"));

    auto semi = o_filt.left_semi_join(l_filt,
        o_filt.c("o_orderkey") == l_filt.c("l_orderkey"));

    auto agg = semi.aggregate(
        {"o_orderpriority"},
        {{"order_count", AggKind::CountAll, lit(int64_t{0})}});
    return agg.sort({asc("o_orderpriority")}).plan();
}

// ── Q5: Local Supplier Volume ──────────────────────────────────────
//
// Six-way join (customer, orders, lineitem, supplier, nation, region)
// with region name filter and order-date range. Group by n_name.
// Sort desc by revenue.
//
// Join order matters a lot here. Joining rns (region⋈nation⋈supplier)
// against customer on nationkey produces a Cartesian-ish intermediate:
// at SF10 with ASIA's 5 nations, rns has ~20k rows (4k suppliers/
// nation) and customer has ~60k rows/nation, so rns ⋈ customer on
// nationkey is ~1.2B rows — enough to OOM a 94 GiB pool. Instead we
// build the two sides of the foreign-key chain independently:
//
//   dim side: region ⋈ nation ⋈ supplier           (20k rows)
//   fact side: customer ⋈ orders_filt ⋈ lineitem   (~9M at SF10)
//
// and bridge them with a multi-key equi-join on
// (l_suppkey == s_suppkey) ∧ (c_nationkey == s_nationkey). G1-rule
// splits that into single-key on suppkey (supplier PK, so the
// intermediate is bounded by the fact side) plus a post-filter on
// the nationkey equality. Peak intermediate at SF10 ≈ 1.8M rows,
// three orders of magnitude below the old plan.
PlanPtr build_q5(const QueryBuilder& qb) {
    auto c = qb.scan("customer", {"c_custkey", "c_nationkey"});
    auto o = qb.scan("orders",
        {"o_orderkey", "o_custkey", "o_orderdate"});
    auto l = qb.scan("lineitem",
        {"l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"});
    auto s = qb.scan("supplier", {"s_suppkey", "s_nationkey"});
    auto n = qb.scan("nation", {"n_nationkey", "n_name", "n_regionkey"});
    auto r = qb.scan("region", {"r_regionkey", "r_name"});

    auto r_filt = r.filter(
        string_op(StringOpKind::Equal, r.c("r_name"), {"ASIA"}));
    auto o_filt = o.filter(
            (o.c("o_orderdate") >= lit(19940101))
         && (o.c("o_orderdate") <  lit(19950101)));

    // Dimension side: region ⋈ nation ⋈ supplier filtered to ASIA.
    auto rn  = r_filt.inner_join(n,
        r_filt.c("r_regionkey") == n.c("n_regionkey"));
    auto rns = rn.inner_join(s, rn.c("n_nationkey") == s.c("s_nationkey"));

    // Fact side: customer ⋈ orders_filt ⋈ lineitem. c_custkey is
    // unique on customer; o_orderkey is unique on orders — both join
    // outputs stay close to the smaller side's cardinality.
    auto co  = c.inner_join(o_filt,
        c.c("c_custkey") == o_filt.c("o_custkey"));
    auto col = co.inner_join(l,
        co.c("o_orderkey") == l.c("l_orderkey"));

    // Bridge: multi-key equi-join binding the fact side's supplier
    // and its customer's nation to the dimension side's Asian
    // supplier's nation. G1-rule splits into suppkey-first + post-
    // filter on nationkey.
    auto joined = col.inner_join(rns,
            (col.c("l_suppkey")   == rns.c("s_suppkey"))
         && (col.c("c_nationkey") == rns.c("s_nationkey")));

    auto projected = joined.project({
        {"n_name", joined.c("n_name")},
        {"disc_price",
            joined.c("l_extendedprice") * (lit(1.0) - joined.c("l_discount"))},
    });
    auto agg = projected.aggregate(
        {"n_name"},
        {{"revenue", AggKind::Sum, projected.c("disc_price")}});
    return agg.sort({desc("revenue")}).plan();
}

// ── Q6: Forecasting Revenue Change ─────────────────────────────────
PlanPtr build_q6(const QueryBuilder& qb) {
    auto li = qb.scan("lineitem",
        {"l_shipdate", "l_extendedprice", "l_discount", "l_quantity"});
    auto plan = li
        .filter(
            li.c("l_shipdate") >= lit(19940101)
         && li.c("l_shipdate") <  lit(19950101)
         && li.c("l_discount") >= lit(0.05)
         && li.c("l_discount") <= lit(0.07)
         && li.c("l_quantity") <  lit(24.0))
        .project({
            {"revenue", li.c("l_extendedprice") * li.c("l_discount")},
        });
    return plan.aggregate(
        /*keys=*/{},
        {{"total", AggKind::Sum, plan.c("revenue")}}).plan();
}

// ── Q7: Volume Shipping ────────────────────────────────────────────
//
// Self-join on nation (aliased n1, n2). Extract year from l_shipdate.
// Group by (supp_nation, cust_nation, l_year).
PlanPtr build_q7(const QueryBuilder& qb) {
    auto s = qb.scan("supplier", {"s_suppkey", "s_nationkey"});
    auto l = qb.scan("lineitem",
        {"l_orderkey", "l_suppkey", "l_extendedprice", "l_discount", "l_shipdate"});
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey"});
    auto c = qb.scan("customer", {"c_custkey", "c_nationkey"});
    auto n1 = qb.scan("nation", {"n_nationkey", "n_name"});
    auto n2 = qb.scan("nation", {"n_nationkey", "n_name"});

    // Two ways through the country pair: (FRANCE, GERMANY) or (GERMANY, FRANCE).
    auto n1_filt = n1.filter(
        string_op(StringOpKind::In, n1.c("n_name"), {"FRANCE", "GERMANY"}));
    auto n2_filt = n2.filter(
        string_op(StringOpKind::In, n2.c("n_name"), {"FRANCE", "GERMANY"}));

    auto l_filt = l.filter(
            (l.c("l_shipdate") >= lit(19950101))
         && (l.c("l_shipdate") <= lit(19961231)));

    // n1 ⋈ supplier  (supp side). Build = n1_filt (≤25 rows);
    // probe = supplier (10K). Inverted form (supplier ⋈ n1_filt)
    // would put the larger side as build, and partition_hash_join
    // chokes on the resulting 6K-row buckets when partitioning on
    // s_nationkey (25 distinct values).
    auto s_n1 = n1_filt.inner_join(s,
        n1_filt.c("n_nationkey") == s.c("s_nationkey"));
    // n2 ⋈ customer (cust side). Same reasoning — c_nationkey has
    // 25 distinct values, so the 150K-row customer side hashed as
    // build overflows the per-partition cap of 4096.
    auto c_n2 = n2_filt.inner_join(c,
        n2_filt.c("n_nationkey") == c.c("c_nationkey"));
    // supplier-side ⋈ lineitem (on supplier)
    auto sl = s_n1.inner_join(l_filt,
        s_n1.c("s_suppkey") == l_filt.c("l_suppkey"));
    // then ⋈ orders
    auto slo = sl.inner_join(o,
        sl.c("l_orderkey") == o.c("o_orderkey"));
    // then ⋈ (customer ⋈ n2) on o_custkey == c_custkey
    auto joined = slo.inner_join(c_n2,
        slo.c("o_custkey") == c_n2.c("c_custkey"));

    // Post-join: keep only the required-pair rows via OR filter.
    // n1.n_name is the 'supp_nation'; n2.n_name is the 'cust_nation'.
    auto paired = joined.filter(
        (string_op(StringOpKind::Equal, joined.c("n_name"),
                   {"FRANCE"})
            && string_op(StringOpKind::Equal,
                         // second "n_name" in the joined schema:
                         // the customer-side nation alias. By-name
                         // lookup returns the first occurrence, so we
                         // use a fresh ColRef via the original n2
                         // builder which is still live.
                         col_ref(n2_filt.schema()[1].id, t_str, false),
                         {"GERMANY"}))
     || (string_op(StringOpKind::Equal, joined.c("n_name"),
                   {"GERMANY"})
            && string_op(StringOpKind::Equal,
                         col_ref(n2_filt.schema()[1].id, t_str, false),
                         {"FRANCE"})));

    auto projected = paired.project({
        {"supp_nation", paired.c("n_name")},  // first occurrence = n1 side
        {"cust_nation",
            col_ref(n2_filt.schema()[1].id, t_str, false)},
        {"l_year",      year_of(paired.c("l_shipdate"))},
        {"volume",
            paired.c("l_extendedprice") * (lit(1.0) - paired.c("l_discount"))},
    });
    auto agg = projected.aggregate(
        {"supp_nation", "cust_nation", "l_year"},
        {{"revenue", AggKind::Sum, projected.c("volume")}});
    return agg.sort({asc("supp_nation"), asc("cust_nation"), asc("l_year")}).plan();
}

// ── Q8: National Market Share ──────────────────────────────────────
//
// 7 tables, 6 joins, case-when inside an outer aggregate, ratio of
// sums. Simplified in this IR port to the shape of the sums; the final
// ratio would be a post-aggregate project (not shown — trivial once
// the sums exist).
PlanPtr build_q8(const QueryBuilder& qb) {
    auto p   = qb.scan("part",     {"p_partkey", "p_type"});
    auto s   = qb.scan("supplier", {"s_suppkey", "s_nationkey"});
    auto l   = qb.scan("lineitem",
        {"l_orderkey", "l_partkey", "l_suppkey",
         "l_extendedprice", "l_discount"});
    auto o   = qb.scan("orders",   {"o_orderkey", "o_custkey", "o_orderdate"});
    auto c   = qb.scan("customer", {"c_custkey", "c_nationkey"});
    auto n1  = qb.scan("nation",   {"n_nationkey", "n_regionkey"});
    auto n2  = qb.scan("nation",   {"n_nationkey", "n_name"});
    auto r   = qb.scan("region",   {"r_regionkey", "r_name"});

    auto r_filt = r.filter(
        string_op(StringOpKind::Equal, r.c("r_name"), {"AMERICA"}));
    auto p_filt = p.filter(
        string_op(StringOpKind::Equal, p.c("p_type"),
                  {"ECONOMY ANODIZED STEEL"}));
    auto o_filt = o.filter(
            (o.c("o_orderdate") >= lit(19950101))
         && (o.c("o_orderdate") <= lit(19961231)));

    // region ⋈ n1 ⋈ customer ⋈ orders ⋈ lineitem ⋈ part,
    // with supplier ⋈ n2 joined on s_nationkey.
    auto r_n1   = r_filt.inner_join(n1,
        r_filt.c("r_regionkey") == n1.c("n_regionkey"));
    auto r_n1_c = r_n1.inner_join(c,
        r_n1.c("n_nationkey") == c.c("c_nationkey"));
    auto step4  = r_n1_c.inner_join(o_filt,
        r_n1_c.c("c_custkey") == o_filt.c("o_custkey"));
    auto step5  = step4.inner_join(l,
        step4.c("o_orderkey") == l.c("l_orderkey"));
    auto step6  = step5.inner_join(p_filt,
        step5.c("l_partkey") == p_filt.c("p_partkey"));
    auto step7  = step6.inner_join(s,
        step6.c("l_suppkey") == s.c("s_suppkey"));
    auto joined = step7.inner_join(n2,
        step7.c("s_nationkey") == n2.c("n_nationkey"));

    auto projected = joined.project({
        {"o_year", year_of(joined.c("o_orderdate"))},
        {"volume",
            joined.c("l_extendedprice") * (lit(1.0) - joined.c("l_discount"))},
        {"nation", joined.c("n_name")},
    });
    // Pre-split the StringOp (nation = 'BRAZIL') into a BOOL8 column so
    // the subsequent CASE body is purely numeric — same pattern Q12
    // and Q14 use.
    auto with_is_brazil = projected.project({
        {"o_year", projected.c("o_year")},
        {"volume", projected.c("volume")},
        {"is_brazil",
            string_op(StringOpKind::Equal,
                      projected.c("nation"), {"BRAZIL"})},
    });
    // sum(case when nation = 'BRAZIL' then volume else 0) / sum(volume)
    auto with_case = with_is_brazil.project({
        {"o_year", with_is_brazil.c("o_year")},
        {"volume", with_is_brazil.c("volume")},
        {"brazil_volume",
            case_when(
                with_is_brazil.c("is_brazil"),
                with_is_brazil.c("volume"),
                lit(0.0))},
    });
    auto agg = with_case.aggregate(
        {"o_year"},
        {
            {"sum_brazil", AggKind::Sum, with_case.c("brazil_volume")},
            {"sum_total",  AggKind::Sum, with_case.c("volume")},
        });
    // The ratio is computed in a final project.
    auto share = agg.project({
        {"o_year",    agg.c("o_year")},
        {"mkt_share", agg.c("sum_brazil") / agg.c("sum_total")},
    });
    return share.sort({asc("o_year")}).plan();
}

// ── Q9: Product Type Profit Measure ────────────────────────────────
//
// part ⋈ supplier ⋈ lineitem ⋈ partsupp ⋈ orders ⋈ nation (5 joins);
// LIKE filter on p_name; amount = price*(1-disc) - supplycost*qty;
// group by (nation, year), sort by nation, year desc.
PlanPtr build_q9(const QueryBuilder& qb) {
    auto p  = qb.scan("part",     {"p_partkey", "p_name"});
    auto s  = qb.scan("supplier", {"s_suppkey", "s_nationkey"});
    auto l  = qb.scan("lineitem",
        {"l_orderkey", "l_partkey", "l_suppkey",
         "l_quantity", "l_extendedprice", "l_discount"});
    auto ps = qb.scan("partsupp",
        {"ps_partkey", "ps_suppkey", "ps_supplycost"});
    auto o  = qb.scan("orders",   {"o_orderkey", "o_orderdate"});
    auto n  = qb.scan("nation",   {"n_nationkey", "n_name"});

    auto p_filt = p.filter(
        string_op(StringOpKind::Contains, p.c("p_name"), {"green"}));

    // p ⋈ l (on partkey) — large fact-side join.
    auto pl = p_filt.inner_join(l,
        p_filt.c("p_partkey") == l.c("l_partkey"));
    // ⋈ supplier (on l_suppkey == s_suppkey)
    auto pls = pl.inner_join(s,
        pl.c("l_suppkey") == s.c("s_suppkey"));
    // ⋈ partsupp (on ps_partkey == l_partkey AND ps_suppkey == l_suppkey)
    auto plsps = pls.inner_join(ps,
            (pls.c("l_partkey") == ps.c("ps_partkey"))
         && (pls.c("l_suppkey") == ps.c("ps_suppkey")));  // multi-key
    // ⋈ orders
    auto plspso = plsps.inner_join(o,
        plsps.c("l_orderkey") == o.c("o_orderkey"));
    // nation ⋈ plspso. Build = nation (25 rows); probe = plspso
    // (the large running join). With s_nationkey having 25 distinct
    // values, the inverted form would put 14K-row buckets as build
    // and overflow partition_hash_join's per-bucket cap.
    auto joined = n.inner_join(plspso,
        n.c("n_nationkey") == plspso.c("s_nationkey"));

    auto projected = joined.project({
        {"nation", joined.c("n_name")},
        {"o_year", year_of(joined.c("o_orderdate"))},
        {"amount",
            joined.c("l_extendedprice") * (lit(1.0) - joined.c("l_discount"))
          - joined.c("ps_supplycost") * joined.c("l_quantity")},
    });
    auto agg = projected.aggregate(
        {"nation", "o_year"},
        {{"sum_profit", AggKind::Sum, projected.c("amount")}});
    return agg.sort({asc("nation"), desc("o_year")}).plan();
}

// ── Q10: Returned Item Reporting ───────────────────────────────────
PlanPtr build_q10(const QueryBuilder& qb) {
    auto c = qb.scan("customer",
        {"c_custkey", "c_name", "c_address", "c_nationkey",
         "c_phone", "c_acctbal", "c_comment"});
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey", "o_orderdate"});
    auto l = qb.scan("lineitem",
        {"l_orderkey", "l_returnflag", "l_extendedprice", "l_discount"});
    auto n = qb.scan("nation", {"n_nationkey", "n_name"});

    auto o_filt = o.filter(
            (o.c("o_orderdate") >= lit(19931001))
         && (o.c("o_orderdate") <  lit(19940101)));
    auto l_filt = l.filter(
        string_op(StringOpKind::Equal, l.c("l_returnflag"), {"R"}));

    auto co  = c.inner_join(o_filt,
        c.c("c_custkey") == o_filt.c("o_custkey"));
    auto col = co.inner_join(l_filt,
        co.c("o_orderkey") == l_filt.c("l_orderkey"));
    // nation ⋈ col. Build = nation (25 rows); probe = col (the
    // big customer⋈orders⋈lineitem result). Inverted form
    // (col.inner_join(n)) overflows partition_hash_join: c_nationkey
    // has 25 distinct values, so the col side hashed as build would
    // give ~5K-row buckets exceeding the 4096 cap.
    auto joined = n.inner_join(col,
        n.c("n_nationkey") == col.c("c_nationkey"));

    auto projected = joined.project({
        {"c_custkey",  joined.c("c_custkey")},
        {"c_name",     joined.c("c_name")},
        {"c_acctbal",  joined.c("c_acctbal")},
        {"c_phone",    joined.c("c_phone")},
        {"n_name",     joined.c("n_name")},
        {"c_address",  joined.c("c_address")},
        {"c_comment",  joined.c("c_comment")},
        {"disc_price",
            joined.c("l_extendedprice") * (lit(1.0) - joined.c("l_discount"))},
    });
    auto agg = projected.aggregate(
        {"c_custkey","c_name","c_acctbal","c_phone","n_name",
         "c_address","c_comment"},
        {{"revenue", AggKind::Sum, projected.c("disc_price")}});
    // Spec output order is (c_custkey, c_name, revenue, c_acctbal,
    // n_name, c_address, c_phone, c_comment); the aggregate emits
    // keys-first (revenue last). Pure-ColRef Project reorders
    // without renumbering ids — the Sort's reference to `revenue`
    // still resolves.
    auto out = agg.project({
        {"c_custkey", agg.c("c_custkey")},
        {"c_name",    agg.c("c_name")},
        {"revenue",   agg.c("revenue")},
        {"c_acctbal", agg.c("c_acctbal")},
        {"n_name",    agg.c("n_name")},
        {"c_address", agg.c("c_address")},
        {"c_phone",   agg.c("c_phone")},
        {"c_comment", agg.c("c_comment")},
    });
    return out.sort({desc("revenue")}).limit(20).plan();
}

// ── Q11: Important Stock Identification ────────────────────────────
//
// partsupp ⋈ supplier ⋈ nation, filtered by nation name. Group by
// ps_partkey summing value. The HAVING clause compares the group sum
// against a scalar threshold derived from a sibling aggregate over the
// same filtered join — Maximus lowers this as a cross_join between the
// grouped output and the 1-row threshold, followed by a filter.
PlanPtr build_q11(const QueryBuilder& qb) {
    auto ps = qb.scan("partsupp",
        {"ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost"});
    auto s  = qb.scan("supplier", {"s_suppkey", "s_nationkey"});
    auto n  = qb.scan("nation",   {"n_nationkey", "n_name"});

    auto n_filt = n.filter(
        string_op(StringOpKind::Equal, n.c("n_name"), {"GERMANY"}));
    auto sn = s.inner_join(n_filt,
        s.c("s_nationkey") == n_filt.c("n_nationkey"));
    auto pssn = ps.inner_join(sn,
        ps.c("ps_suppkey") == sn.c("s_suppkey"));

    auto with_value = pssn.project({
        {"ps_partkey", pssn.c("ps_partkey")},
        {"value", pssn.c("ps_supplycost") * pssn.c("ps_availqty")},
    });

    // Main grouped aggregate.
    auto grouped = with_value.aggregate(
        {"ps_partkey"},
        {{"partkey_value", AggKind::Sum, with_value.c("value")}});

    // Threshold: sum(value) over the same filtered join * 0.0001.
    auto threshold_agg = with_value.aggregate(
        /*keys=*/{},
        {{"global_sum", AggKind::Sum, with_value.c("value")}});
    auto threshold = threshold_agg.project({
        {"threshold", threshold_agg.c("global_sum") * lit(0.0001)},
    });

    // Cross-join the per-partkey sums with the 1-row threshold, then
    // filter. (Physical emit of cross_join is not wired yet.)
    auto crossed = grouped.cross_join(threshold);
    auto filtered = crossed.filter(
        crossed.c("partkey_value") > crossed.c("threshold"));

    auto projected = filtered.project({
        {"ps_partkey",    filtered.c("ps_partkey")},
        {"partkey_value", filtered.c("partkey_value")},
    });
    return projected.sort({desc("partkey_value")}).plan();
}

// ── Q12: Shipping Modes and Order Priority ─────────────────────────
//
// CASE-in-aggregate pattern: count lineitems by shipmode, split into
// "high priority" (orderpriority ∈ {URGENT, HIGH}) and "low priority"
// buckets. IR expresses the CASE inside a Project so each aggregate
// body is a pure ColRef.
PlanPtr build_q12(const QueryBuilder& qb) {
    auto o = qb.scan("orders", {"o_orderkey", "o_orderpriority"});
    auto l = qb.scan("lineitem",
        {"l_orderkey", "l_shipmode", "l_commitdate",
         "l_receiptdate", "l_shipdate"});

    auto l_filt = l.filter(
            string_op(StringOpKind::In, l.c("l_shipmode"),
                      {"MAIL", "SHIP"})
         && (l.c("l_commitdate") < l.c("l_receiptdate"))
         && (l.c("l_shipdate")   < l.c("l_commitdate"))
         && (l.c("l_receiptdate") >= lit(19940101))
         && (l.c("l_receiptdate") <  lit(19950101)));
    auto joined = o.inner_join(l_filt,
        o.c("o_orderkey") == l_filt.c("l_orderkey"));

    // First Project materialises `is_high` as a BOOL8 column — the
    // top-level-StringOp-as-Project-item path supported by gap #10.
    // Second Project's CASE bodies then use it as a pure ColRef, so
    // the StringOp never appears inside a computational ExprTree.
    auto with_is_high = joined.project({
        {"l_shipmode", joined.c("l_shipmode")},
        {"is_high",
            string_op(StringOpKind::Equal, joined.c("o_orderpriority"),
                      {"1-URGENT"})
         || string_op(StringOpKind::Equal, joined.c("o_orderpriority"),
                      {"2-HIGH"})},
    });
    auto with_flags = with_is_high.project({
        {"l_shipmode", with_is_high.c("l_shipmode")},
        {"high",
            case_when(with_is_high.c("is_high"),
                      lit(int64_t{1}), lit(int64_t{0}))},
        {"low",
            case_when(with_is_high.c("is_high"),
                      lit(int64_t{0}), lit(int64_t{1}))},
    });
    auto agg = with_flags.aggregate(
        {"l_shipmode"},
        {
            {"high_line_count", AggKind::Sum, with_flags.c("high")},
            {"low_line_count",  AggKind::Sum, with_flags.c("low")},
        });
    return agg.sort({asc("l_shipmode")}).plan();
}

// ── Q13: Customer Distribution ─────────────────────────────────────
//
// Canonical Q13 has a LEFT OUTER JOIN with a filter embedded in its
// ON clause — each customer survives with c_count=0 if no matching
// order exists. gamax's phys layer has no outer join and no NULLs,
// so we rewrite at the builder level:
//
//   orders_clean = orders where NOT LIKE '%special%requests%'  (RHS-only
//                  filter → safe to hoist out of the outer-join ON
//                  for LEFT OUTER; DuckDB does the same rewrite.)
//
//   matched = customer ⋈_inner orders_clean on c_custkey == o_custkey
//           -> group by c_custkey, count_all → per-customer c_count
//
//   unmatched = customer ⟕_anti orders_clean on c_custkey == o_custkey
//             -> project(c_custkey, c_count = 0)
//
//   concat(matched, unmatched) → one (c_custkey, c_count) row per
//   customer, with c_count=0 for those who had no qualifying order.
//
//   histogram = group by c_count, count_all as custdist
//
// The aggregation is pushed *below* the concat so the union of the
// two branches is 1.5M rows at SF10 rather than tens of millions.
// See .claude/outer-join-for-q13.md for the design doc.
PlanPtr build_q13(const QueryBuilder& qb) {
    auto c = qb.scan("customer", {"c_custkey"});
    auto o = qb.scan("orders", {"o_custkey", "o_comment"});

    // The spec's LIKE pattern is `%special%requests%` — both tokens
    // in order. `ContainsMultiple` matches the spec exactly; the
    // previous builder used a single-pattern `Contains("special")`
    // which overmatched on SF1 (comments with "special" but not
    // "requests" were wrongly filtered out).
    auto o_clean = o.filter(
        !string_op(StringOpKind::ContainsMultiple, o.c("o_comment"),
                   {"special", "requests"}));

    // Matched branch: inner join + group-by customer → per-customer
    // order count. G4a rewrites CountAll → Sum(lit(1)) at apply_rules
    // time.
    auto matched_join = c.inner_join(o_clean,
        c.c("c_custkey") == o_clean.c("o_custkey"));
    auto matched = matched_join.aggregate(
        {"c_custkey"},
        {{"c_count", AggKind::CountAll, lit(int64_t{0})}});

    // Unmatched branch: anti-join + project a zero c_count column.
    auto unmatched_join = c.left_anti_join(o_clean,
        c.c("c_custkey") == o_clean.c("o_custkey"));
    auto unmatched = unmatched_join.project({
        {"c_custkey", unmatched_join.c("c_custkey")},
        {"c_count",   lit(int64_t{0})},
    });

    auto all_customers = matched.concat(unmatched);

    auto histogram = all_customers.aggregate(
        {"c_count"},
        {{"custdist", AggKind::CountAll, lit(int64_t{0})}});
    return histogram.sort({desc("custdist"), desc("c_count")}).plan();
}

// ── Q14: Promotion Effect ──────────────────────────────────────────
PlanPtr build_q14(const QueryBuilder& qb) {
    auto l = qb.scan("lineitem",
        {"l_orderkey", "l_partkey", "l_extendedprice",
         "l_discount", "l_shipdate"});
    auto p = qb.scan("part", {"p_partkey", "p_type"});

    auto l_filt = l.filter(
            (l.c("l_shipdate") >= lit(19950901))
         && (l.c("l_shipdate") <  lit(19951001)));
    auto joined = l_filt.inner_join(p,
        l_filt.c("l_partkey") == p.c("p_partkey"));

    auto with_classification = joined.project({
        {"disc_price",
            joined.c("l_extendedprice") *
            (lit(1.0) - joined.c("l_discount"))},
        {"is_promo",
            string_op(StringOpKind::Startswith,
                      joined.c("p_type"), {"PROMO"})},
    });
    auto with_case = with_classification.project({
        {"promo_volume",
            case_when(with_classification.c("is_promo"),
                      with_classification.c("disc_price"),
                      lit(0.0))},
        {"volume", with_classification.c("disc_price")},
    });
    auto agg = with_case.aggregate(
        /*keys=*/{},
        {
            {"sum_promo", AggKind::Sum, with_case.c("promo_volume")},
            {"sum_total", AggKind::Sum, with_case.c("volume")},
        });
    return agg.project({
        {"promo_revenue",
            lit(100.0) * agg.c("sum_promo") / agg.c("sum_total")},
    }).plan();
}

// ── Q15: Top Supplier Query ────────────────────────────────────────
//
// Three-phase: (1) per-supplier revenue within a date window;
// (2) scalar max of those revenues; (3) cross-join the revenue list
// with the max scalar and keep rows where revenue == max, then join
// with supplier for the output. Maximus does the same shape.
PlanPtr build_q15(const QueryBuilder& qb) {
    auto l = qb.scan("lineitem",
        {"l_suppkey", "l_extendedprice", "l_discount", "l_shipdate"});
    auto l_filt = l.filter(
            (l.c("l_shipdate") >= lit(19960101))
         && (l.c("l_shipdate") <  lit(19960401)));
    auto revenue_body = l_filt.project({
        {"supplier_no", l_filt.c("l_suppkey")},
        {"rev", l_filt.c("l_extendedprice") *
                (lit(1.0) - l_filt.c("l_discount"))},
    });
    auto revenue = revenue_body.aggregate(
        {"supplier_no"},
        {{"total_revenue", AggKind::Sum, revenue_body.c("rev")}});

    auto max_rev = revenue.aggregate(
        /*keys=*/{},
        {{"max_revenue", AggKind::Max, revenue.c("total_revenue")}});

    auto crossed = revenue.cross_join(max_rev);
    auto best = crossed.filter(
        crossed.c("total_revenue") == crossed.c("max_revenue"));

    auto s = qb.scan("supplier",
        {"s_suppkey", "s_name", "s_address", "s_phone"});
    auto joined = s.inner_join(best,
        s.c("s_suppkey") == best.c("supplier_no"));
    auto projected = joined.project({
        {"s_suppkey",     joined.c("s_suppkey")},
        {"s_name",        joined.c("s_name")},
        {"s_address",     joined.c("s_address")},
        {"s_phone",       joined.c("s_phone")},
        {"total_revenue", joined.c("total_revenue")},
    });
    return projected.sort({asc("s_suppkey")}).plan();
}

// ── Q16: Parts/Supplier Relationship ───────────────────────────────
//
// NOT IN (subquery) → left_anti_join. COUNT DISTINCT is an IR
// aggregate kind; its physical path isn't wired yet but the plan
// builds.
PlanPtr build_q16(const QueryBuilder& qb) {
    auto ps = qb.scan("partsupp", {"ps_partkey", "ps_suppkey"});
    auto p  = qb.scan("part",
        {"p_partkey", "p_brand", "p_type", "p_size"});
    auto s  = qb.scan("supplier", {"s_suppkey", "s_comment"});

    auto p_filt = p.filter(
            !string_op(StringOpKind::Equal, p.c("p_brand"), {"Brand#45"})
         && !string_op(StringOpKind::Startswith,
                       p.c("p_type"), {"MEDIUM POLISHED"})
         && ( (p.c("p_size") == lit(49))
           || (p.c("p_size") == lit(14))
           || (p.c("p_size") == lit(23))
           || (p.c("p_size") == lit(45))
           || (p.c("p_size") == lit(19))
           || (p.c("p_size") == lit(3))
           || (p.c("p_size") == lit(36))
           || (p.c("p_size") == lit(9)) ));
    // Spec predicate is `s_comment LIKE '%Customer%Complaints%'`.
    // That's sequenced — "Customer" must appear before "Complaints"
    // — which a simple `Contains("Customer") && Contains("Complaints")`
    // doesn't enforce (a comment containing the two substrings in
    // the wrong order would false-match). `ContainsMultiple` lowers
    // to `gamax::contains_multiple`, whose kernel scans patterns
    // left-to-right non-overlapping, which is exactly LIKE's shape.
    auto s_bad = s.filter(
        string_op(StringOpKind::ContainsMultiple, s.c("s_comment"),
                  {"Customer", "Complaints"}));

    // partsupp LEFT ANTI JOIN suppliers-with-bad-comments on ps_suppkey.
    auto ps_clean = ps.left_anti_join(s_bad,
        ps.c("ps_suppkey") == s_bad.c("s_suppkey"));
    auto joined = p_filt.inner_join(ps_clean,
        p_filt.c("p_partkey") == ps_clean.c("ps_partkey"));

    auto agg = joined.aggregate(
        {"p_brand", "p_type", "p_size"},
        {{"supplier_cnt", AggKind::CountDistinct,
          joined.c("ps_suppkey")}});
    return agg.sort({
        desc("supplier_cnt"), asc("p_brand"), asc("p_type"), asc("p_size"),
    }).plan();
}

// ── Q17: Small-Quantity-Order Revenue ──────────────────────────────
//
// Correlated subquery: per-part average quantity threshold, joined back
// to the outer plan on l_partkey. This is the same general pattern as
// Q2's min-supplycost correlation.
PlanPtr build_q17(const QueryBuilder& qb) {
    auto p = qb.scan("part", {"p_partkey", "p_brand", "p_container"});
    auto l = qb.scan("lineitem",
        {"l_orderkey", "l_partkey", "l_quantity", "l_extendedprice"});
    auto p_filt = p.filter(
            string_op(StringOpKind::Equal, p.c("p_brand"), {"Brand#23"})
         && string_op(StringOpKind::Equal, p.c("p_container"), {"MED BOX"}));
    auto pl = p_filt.inner_join(l,
        p_filt.c("p_partkey") == l.c("l_partkey"));

    // Subquery: avg quantity per partkey across all of lineitem.
    auto l2 = qb.scan("lineitem", {"l_partkey", "l_quantity"});
    auto per_part = l2.aggregate(
        {"l_partkey"},
        {{"avg_qty", AggKind::Avg, l2.c("l_quantity")}});
    auto threshold = per_part.project({
        {"l_partkey", per_part.c("l_partkey")},
        {"thr", lit(0.2) * per_part.c("avg_qty")},
    });

    auto with_thresh = pl.inner_join(threshold,
        pl.c("l_partkey") == threshold.c("l_partkey"));
    auto below = with_thresh.filter(
        with_thresh.c("l_quantity") < with_thresh.c("thr"));
    auto agg = below.aggregate(
        /*keys=*/{},
        {{"sum_price", AggKind::Sum, below.c("l_extendedprice")}});
    return agg.project({
        {"avg_yearly", agg.c("sum_price") / lit(7.0)},
    }).plan();
}

// ── Q18: Large Volume Customer ─────────────────────────────────────
//
// The IN-subquery over orders whose lineitem sum(quantity) > 300 is
// lowered as: (1) aggregate lineitem per orderkey; (2) filter big
// orders; (3) left_semi_join orders against the big-orders plan.
PlanPtr build_q18(const QueryBuilder& qb) {
    auto l_sub = qb.scan("lineitem", {"l_orderkey", "l_quantity"});
    auto big_orders_agg = l_sub.aggregate(
        {"l_orderkey"},
        {{"sum_qty", AggKind::Sum, l_sub.c("l_quantity")}});
    auto big = big_orders_agg.filter(
        big_orders_agg.c("sum_qty") > lit(300.0));

    auto c = qb.scan("customer", {"c_custkey", "c_name"});
    auto o = qb.scan("orders",
        {"o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"});
    auto l = qb.scan("lineitem", {"l_orderkey", "l_quantity"});

    auto o_big = o.left_semi_join(big,
        o.c("o_orderkey") == big.c("l_orderkey"));
    auto co  = c.inner_join(o_big,
        c.c("c_custkey") == o_big.c("o_custkey"));
    auto col = co.inner_join(l,
        co.c("o_orderkey") == l.c("l_orderkey"));
    auto agg = col.aggregate(
        {"c_name", "c_custkey", "o_orderkey",
         "o_orderdate", "o_totalprice"},
        {{"sum_quantity", AggKind::Sum, col.c("l_quantity")}});
    return agg.sort({desc("o_totalprice"), asc("o_orderdate")})
              .limit(100).plan();
}

// ── Q19: Discounted Revenue ────────────────────────────────────────
//
// Three disjunctive branches over brand/container/qty/size with a
// shared shipmode+shipinstruct prefix. The shared prefix is highly
// selective (~1% of lineitem) and depends only on lineitem columns,
// so we apply it before the join — shrinking the build/probe input
// and dropping the string columns from the join's gather payload.
// The three brand/container/qty/size branches still need post-join
// columns (p_brand, p_container, p_size + l_quantity), so they stay
// as a post-join filter.
PlanPtr build_q19(const QueryBuilder& qb) {
    auto l = qb.scan("lineitem",
        {"l_partkey", "l_quantity", "l_extendedprice", "l_discount",
         "l_shipmode", "l_shipinstruct"});
    auto l_filt = l.filter(
            string_op(StringOpKind::In, l.c("l_shipmode"),
                      {"AIR", "AIR REG"})
         && string_op(StringOpKind::Equal, l.c("l_shipinstruct"),
                      {"DELIVER IN PERSON"}));
    // Drop the string columns that only the pre-filter needed; the
    // join + post-filter only need the four numeric/key columns.
    auto l_proj = l_filt.project({
        {"l_partkey",       l_filt.c("l_partkey")},
        {"l_quantity",      l_filt.c("l_quantity")},
        {"l_extendedprice", l_filt.c("l_extendedprice")},
        {"l_discount",      l_filt.c("l_discount")},
    });

    auto p = qb.scan("part",
        {"p_partkey", "p_brand", "p_container", "p_size"});
    auto j = l_proj.inner_join(p, l_proj.c("l_partkey") == p.c("p_partkey"));

    auto branch = [&](const char* brand,
                      std::vector<std::string> containers,
                      double q_lo, double q_hi,
                      int32_t sz_lo, int32_t sz_hi) {
        return string_op(StringOpKind::Equal, j.c("p_brand"), {brand})
            && string_op(StringOpKind::In,
                         j.c("p_container"), std::move(containers))
            && (j.c("l_quantity") >= lit(q_lo))
            && (j.c("l_quantity") <= lit(q_hi))
            && (j.c("p_size") >= lit(sz_lo))
            && (j.c("p_size") <= lit(sz_hi));
    };
    auto filter =
            branch("Brand#12",
                   {"SM CASE","SM BOX","SM PACK","SM PKG"}, 1.0, 11.0, 1, 5)
         || branch("Brand#23",
                   {"MED BAG","MED BOX","MED PKG","MED PACK"},
                   10.0, 20.0, 1, 10)
         || branch("Brand#34",
                   {"LG CASE","LG BOX","LG PACK","LG PKG"},
                   20.0, 30.0, 1, 15);
    auto filtered = j.filter(std::move(filter));
    auto projected = filtered.project({
        {"disc_price",
            filtered.c("l_extendedprice") *
            (lit(1.0) - filtered.c("l_discount"))},
    });
    return projected.aggregate(
        /*keys=*/{},
        {{"revenue", AggKind::Sum, projected.c("disc_price")}}).plan();
}

// ── Q20: Potential Part Promotion ──────────────────────────────────
//
// Deep nested IN / correlated threshold: select suppliers (in a given
// nation) whose partsupp entries for "forest" parts have availqty >
// 0.5 * sum(shipdate-bounded lineitem qty). Lowers to: semi-joins for
// the IN, inner join against the threshold table (multi-key), then
// another semi-join to prune suppliers.
PlanPtr build_q20(const QueryBuilder& qb) {
    auto p = qb.scan("part", {"p_partkey", "p_name"});
    // Spec is `p_name LIKE 'forest%'` (startswith), not contains.
    // Using Contains overmatches and roughly doubled the row count.
    auto p_filt = p.filter(
        string_op(StringOpKind::Startswith, p.c("p_name"), {"forest"}));

    auto l_sub = qb.scan("lineitem",
        {"l_partkey", "l_suppkey", "l_quantity", "l_shipdate"});
    auto l_filt = l_sub.filter(
            (l_sub.c("l_shipdate") >= lit(19940101))
         && (l_sub.c("l_shipdate") <  lit(19950101)));
    auto qty = l_filt.aggregate(
        {"l_partkey", "l_suppkey"},
        {{"sum_qty", AggKind::Sum, l_filt.c("l_quantity")}});
    auto threshold = qty.project({
        {"l_partkey", qty.c("l_partkey")},
        {"l_suppkey", qty.c("l_suppkey")},
        {"thr", lit(0.5) * qty.c("sum_qty")},
    });

    auto ps = qb.scan("partsupp",
        {"ps_partkey", "ps_suppkey", "ps_availqty"});
    auto ps_in_p = ps.left_semi_join(p_filt,
        ps.c("ps_partkey") == p_filt.c("p_partkey"));
    auto with_thr = ps_in_p.inner_join(threshold,
            (ps_in_p.c("ps_partkey") == threshold.c("l_partkey"))
         && (ps_in_p.c("ps_suppkey") == threshold.c("l_suppkey")));
    auto ps_ok = with_thr.filter(
        with_thr.c("ps_availqty") > with_thr.c("thr"));

    auto s = qb.scan("supplier",
        {"s_suppkey", "s_name", "s_address", "s_nationkey"});
    auto s_ok = s.left_semi_join(ps_ok,
        s.c("s_suppkey") == ps_ok.c("ps_suppkey"));
    auto n = qb.scan("nation", {"n_nationkey", "n_name"});
    auto n_filt = n.filter(
        string_op(StringOpKind::Equal, n.c("n_name"), {"CANADA"}));
    auto joined = s_ok.inner_join(n_filt,
        s_ok.c("s_nationkey") == n_filt.c("n_nationkey"));

    auto projected = joined.project({
        {"s_name",    joined.c("s_name")},
        {"s_address", joined.c("s_address")},
    });
    return projected.sort({asc("s_name")}).plan();
}

// ── Q21: Suppliers Who Kept Orders Waiting ─────────────────────────
//
// Canonical Q21 has two correlated subqueries whose match predicates
// each carry a `<>` non-equi residual against the outer row:
//
//   EXISTS     (l2 | l2.l_orderkey  = l1.l_orderkey
//                 AND l2.l_suppkey <> l1.l_suppkey)
//   NOT EXISTS (l3 | l3.l_orderkey  = l1.l_orderkey
//                 AND l3.l_suppkey <> l1.l_suppkey
//                 AND l3.l_receiptdate > l3.l_commitdate)
//
// gamax's semi / anti joins require pure equi conditions, so
// lowering these directly hits `join: condition must be a
// conjunction of column equi-predicates`. But the `<>` residuals
// are algebraically absorbable: for a fixed l1.l_orderkey, the
// residual "there is some OTHER suppkey on this order" is equivalent
// to "the distinct-suppkey count on this order is > 1", an
// order-wide aggregate condition that does not depend on which
// supplier l1 is. We precompute it once, then the membership check
// becomes a pure equi-semi-join on l_orderkey — the same trick
// Maximus uses (their Q21 in tpch_queries.cpp).
//
// The parallel rewrite for NOT EXISTS: "no OTHER failed supplier on
// this order" ⟺ "all failed lineitems on this order have the same
// suppkey" ⟺ "distinct-suppkey count over FAILED lineitems of this
// order is == 1" (since l1 is itself a failed lineitem, that sole
// suppkey is l1.suppkey).
//
// Two independent `count_distinct(l_suppkey) group by l_orderkey`
// aggregates — one over all lineitem (EXISTS), one over failed
// lineitem (NOT EXISTS) — produce the order-side membership sets,
// and the main `failed` relation is whittled down via two
// equi-semi-joins on l_orderkey. No non-equi anywhere in the plan.
PlanPtr build_q21(const QueryBuilder& qb) {
    auto l_all  = qb.scan("lineitem", {"l_orderkey", "l_suppkey"});
    auto l_main = qb.scan("lineitem",
        {"l_orderkey", "l_suppkey", "l_commitdate", "l_receiptdate"});
    auto s = qb.scan("supplier", {"s_suppkey", "s_name", "s_nationkey"});
    auto o = qb.scan("orders",   {"o_orderkey", "o_orderstatus"});
    auto n = qb.scan("nation",   {"n_nationkey", "n_name"});

    // Failed lineitems (l1_late in the spec).
    auto failed = l_main.filter(
        l_main.c("l_receiptdate") > l_main.c("l_commitdate"));

    // EXISTS rewrite: orders with > 1 distinct supplier.
    auto supp_per_order = l_all.aggregate(
        {"l_orderkey"},
        {{"n_supp", AggKind::CountDistinct, l_all.c("l_suppkey")}});
    auto multi_supp = supp_per_order.filter(
        supp_per_order.c("n_supp") > lit(int64_t{1}));

    // NOT EXISTS rewrite: orders with exactly 1 distinct failed
    // supplier (which — since l1 itself is failed on this order —
    // must be l1.suppkey).
    auto failed_supp_per_order = failed.aggregate(
        {"l_orderkey"},
        {{"n_failed_supp", AggKind::CountDistinct,
          failed.c("l_suppkey")}});
    auto only_one_failed = failed_supp_per_order.filter(
        failed_supp_per_order.c("n_failed_supp") == lit(int64_t{1}));

    // Equi-semi-joins — no non-equi residuals, lowering is happy.
    auto in_multi = failed.left_semi_join(multi_supp,
        failed.c("l_orderkey") == multi_supp.c("l_orderkey"));
    auto surviving = in_multi.left_semi_join(only_one_failed,
        in_multi.c("l_orderkey") == only_one_failed.c("l_orderkey"));

    auto o_f = o.filter(
        string_op(StringOpKind::Equal, o.c("o_orderstatus"), {"F"}));
    auto n_f = n.filter(
        string_op(StringOpKind::Equal, n.c("n_name"), {"SAUDI ARABIA"}));

    auto j1 = surviving.inner_join(s,
        surviving.c("l_suppkey") == s.c("s_suppkey"));
    auto j2 = j1.inner_join(o_f,
        j1.c("l_orderkey") == o_f.c("o_orderkey"));
    // n_f ⋈ j2. Build = nation_filt (1 row, "SAUDI ARABIA"); probe
    // = j2 (the big running join). Inverted form would overflow
    // partition_hash_join because s_nationkey has 25 distinct
    // values across j2's many rows.
    auto j3 = n_f.inner_join(j2,
        n_f.c("n_nationkey") == j2.c("s_nationkey"));

    auto agg = j3.aggregate(
        {"s_name"},
        {{"numwait", AggKind::CountAll, lit(int64_t{0})}});
    return agg.sort({desc("numwait"), asc("s_name")}).limit(100).plan();
}

// ── Q22: Global Sales Opportunity ──────────────────────────────────
//
// substring() via func(), plus the avg-threshold / cross_join pattern
// from Q11, plus a NOT EXISTS → left_anti_join.
PlanPtr build_q22(const QueryBuilder& qb) {
    auto c = qb.scan("customer", {"c_custkey", "c_phone", "c_acctbal"});
    auto with_cc = c.project({
        {"c_custkey", c.c("c_custkey")},
        {"c_acctbal", c.c("c_acctbal")},
        {"cntrycode",
            func("substring",
                 {c.c("c_phone"), lit(int64_t{1}), lit(int64_t{2})},
                 t_str)},
    });
    const std::vector<std::string> cc_set = {
        "13","31","23","29","30","18","17"};
    auto in_set = with_cc.filter(
        string_op(StringOpKind::In, with_cc.c("cntrycode"), cc_set));

    // Threshold plan: avg(c_acctbal) for c_acctbal > 0 within the same
    // cntrycode set.
    auto for_avg = in_set.filter(in_set.c("c_acctbal") > lit(0.0));
    auto avg_plan = for_avg.aggregate(
        /*keys=*/{},
        {{"avg_acctbal", AggKind::Avg, for_avg.c("c_acctbal")}});

    // Cross-join the full in-set with the 1-row threshold, filter by
    // c_acctbal > avg.
    auto crossed = in_set.cross_join(avg_plan);
    auto above = crossed.filter(
        crossed.c("c_acctbal") > crossed.c("avg_acctbal"));

    // NOT EXISTS orders with o_custkey = c_custkey.
    auto o = qb.scan("orders", {"o_custkey"});
    auto no_orders = above.left_anti_join(o,
        above.c("c_custkey") == o.c("o_custkey"));

    auto agg = no_orders.aggregate(
        {"cntrycode"},
        {
            {"numcust",    AggKind::CountAll, lit(int64_t{0})},
            {"totacctbal", AggKind::Sum, no_orders.c("c_acctbal")},
        });
    return agg.sort({asc("cntrycode")}).plan();
}

} // namespace gamax::ir::tpch
