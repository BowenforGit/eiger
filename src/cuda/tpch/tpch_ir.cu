#include "tpch/tpch_ir.hpp"

#include "ir/expr.hpp"
#include "ir/lower_to_exec_plan.hpp"
#include "operators/groupby/groupby.hpp"

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

    // The natural Q1 form: sum + avg both directly. The lowering's
    // materialize_keys_and_values dedups ColRef-bodied agg items by
    // ColumnId, so sum(l_quantity) and avg(l_quantity) share a
    // single materialized body column (saving ~4.4 GiB of duplicate
    // intermediate at SF=100); same for sum(l_extendedprice) /
    // avg(l_extendedprice). count_order goes through the phys
    // layer's native COUNT_ALL — no lit-1 column materialization.
    auto agg = projected.aggregate(
        {"l_returnflag", "l_linestatus"},
        {
            {"sum_qty",        AggKind::Sum, projected.c("l_quantity")},
            {"sum_base_price", AggKind::Sum, projected.c("l_extendedprice")},
            {"sum_disc_price", AggKind::Sum, projected.c("disc_price")},
            {"sum_charge",     AggKind::Sum, projected.c("charge")},
            {"avg_qty",        AggKind::Avg, projected.c("l_quantity")},
            {"avg_price",      AggKind::Avg, projected.c("l_extendedprice")},
            {"avg_disc",       AggKind::Avg, projected.c("l_discount")},
            {"count_order",    AggKind::CountAll, lit(int64_t{0})},
        });

    auto sorted = agg.sort({asc("l_returnflag"), asc("l_linestatus")});
    return sorted.plan();
}

// ── Q2: Minimum Cost Supplier ──────────────────────────────────────
//
// Five-way join (part, supplier, partsupp, nation, region) with size,
// p_type endswith '[type]', and r_name filters. The correlated MIN
// subquery is decorrelated as a per-partkey aggregate over the same
// (region-filtered) join, joined back on
//   (p_partkey, ps_supplycost) == (p_partkey, min_supplycost).
//
// Single-chain shape (DAG): one filtered psps_nr relation is reused
// both as the agg input and as the back-join probe. Pushing part_filt
// above the agg is sound — the agg is keyed on p_partkey, so dropping
// non-matching partkeys before aggregation only drops groups that the
// outer plan would never reference.
//
// ORDER BY s_acctbal DESC, n_name, s_name, p_partkey  LIMIT 100.
PlanPtr build_q2(const QueryBuilder& qb) {
    auto p = qb.scan("part",
        {"p_partkey", "p_mfgr", "p_type", "p_size"});
    auto ps = qb.scan("partsupp",
        {"ps_partkey", "ps_suppkey", "ps_supplycost"});
    auto s = qb.scan("supplier",
        {"s_suppkey", "s_name", "s_address", "s_nationkey",
         "s_phone", "s_acctbal", "s_comment"});
    auto n = qb.scan("nation", {"n_nationkey", "n_name", "n_regionkey"});
    auto r = qb.scan("region", {"r_regionkey", "r_name"});

    auto p_filt = p
        .filter(p.c("p_size") == lit(15))
        .filter(string_op(StringOpKind::Endswith,
                          p.c("p_type"), {"BRASS"}));
    auto p_proj = p_filt.project({
        {"p_partkey", p_filt.c("p_partkey")},
        {"p_mfgr",    p_filt.c("p_mfgr")},
    });

    // Maximus inner_join(A, B): A=probe (1st), B=build (2nd).
    // Our IR right=probe, left=build → swap args.
    auto psp = ps.inner_join(p_proj,
        ps.c("ps_partkey") == p_proj.c("p_partkey"));
    auto psps = s.inner_join(psp,
        s.c("s_suppkey") == psp.c("ps_suppkey"));
    auto psps_proj = psps.project({
        {"p_partkey",     psps.c("p_partkey")},
        {"ps_supplycost", psps.c("ps_supplycost")},
        {"p_mfgr",        psps.c("p_mfgr")},
        {"s_nationkey",   psps.c("s_nationkey")},
        {"s_acctbal",     psps.c("s_acctbal")},
        {"s_name",        psps.c("s_name")},
        {"s_address",     psps.c("s_address")},
        {"s_phone",       psps.c("s_phone")},
        {"s_comment",     psps.c("s_comment")},
    });

    auto r_filt = r.filter(
        string_op(StringOpKind::Equal, r.c("r_name"), {"EUROPE"}));
    auto nr = r_filt.inner_join(n,
        r_filt.c("r_regionkey") == n.c("n_regionkey"));
    auto nr_proj = nr.project({
        {"n_nationkey", nr.c("n_nationkey")},
        {"n_name",      nr.c("n_name")},
    });

    auto psps_nr = nr_proj.inner_join(psps_proj,
        nr_proj.c("n_nationkey") == psps_proj.c("s_nationkey"));
    auto psps_nr_proj = psps_nr.project({
        {"p_partkey",     psps_nr.c("p_partkey")},
        {"ps_supplycost", psps_nr.c("ps_supplycost")},
        {"p_mfgr",        psps_nr.c("p_mfgr")},
        {"n_name",        psps_nr.c("n_name")},
        {"s_acctbal",     psps_nr.c("s_acctbal")},
        {"s_name",        psps_nr.c("s_name")},
        {"s_address",     psps_nr.c("s_address")},
        {"s_phone",       psps_nr.c("s_phone")},
        {"s_comment",     psps_nr.c("s_comment")},
    });

    // DAG reuse: psps_nr_proj feeds both the per-partkey min and the
    // back-join's probe.
    auto agg = psps_nr_proj.aggregate(
        {"p_partkey"},
        {{"min_supplycost",
            AggKind::Min, psps_nr_proj.c("ps_supplycost")}});
    // Rename agg's `p_partkey` so the back-join's `c("p_partkey")` on
    // the result side resolves unambiguously to psps_nr_proj's column.
    auto agg_renamed = agg.project({
        {"agg_partkey",    agg.c("p_partkey")},
        {"min_supplycost", agg.c("min_supplycost")},
    });

    auto final_join = agg_renamed.inner_join(psps_nr_proj,
            (agg_renamed.c("agg_partkey")    == psps_nr_proj.c("p_partkey"))
         && (agg_renamed.c("min_supplycost") == psps_nr_proj.c("ps_supplycost")));

    auto out = final_join.project({
        {"s_acctbal", final_join.c("s_acctbal")},
        {"s_name",    final_join.c("s_name")},
        {"n_name",    final_join.c("n_name")},
        {"p_partkey", final_join.c("p_partkey")},
        {"p_mfgr",    final_join.c("p_mfgr")},
        {"s_address", final_join.c("s_address")},
        {"s_phone",   final_join.c("s_phone")},
        {"s_comment", final_join.c("s_comment")},
    });
    return out.sort({
        desc("s_acctbal"), asc("n_name"), asc("s_name"), asc("p_partkey"),
    }).limit(100).plan();
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

// ── Q71–Q73: alternative Q7 plan (ported from an external IR) ───────
//
// A different join order from build_q7 above: it grows the CUSTOMER
// side down through orders → lineitem first (cn → cno → cnol), then
// joins the SUPPLIER side (sn) last — where build_q7 does the reverse
// (supplier→lineitem→orders→customer). It also projects aggressively
// after every join, carrying only the keys and the two nation names.
//
// gamax mapping notes:
//   * gamax has no builder-level `rename`. The source renames the two
//     nation instances to n1_*/n2_* so downstream name lookup is
//     unambiguous; we reproduce that with a pure-ColRef project (which
//     preserves ColumnIds — the rename is cosmetic). This also avoids
//     the raw `col_ref(n2.schema()[1].id, ...)` the base Q7 needs.
//   * The source spells n2 as `n1.rename(...)`, i.e. a *distinct*
//     nation instance sharing n1's rows. gamax needs distinct
//     ColumnIds per join side, so n1 and n2 are two independent nation
//     scans with the same {FRANCE, GERMANY} filter (same rows, fresh
//     ids) — exactly what base Q7 does.
//   * Build/probe order of the five inner joins is unknown, so each is
//     parameterized by an s1..s5 swap. In gamax `X.inner_join(Y)` puts
//     X on the build (left) side; swap=false keeps the source's caller
//     as build, swap=true builds on the argument instead.
//
// Variants:
//   Q71 (00000): every join as-written (source caller = build)
//   Q72 (11111): every join swapped (source argument = build)
//   Q73 (00111): small-side-build — n1/n2 (2 rows) build joins 1-2;
//                cn/cno/sn (the small filtered CTEs) build joins 3-5.
static PlanPtr build_q7_alt(const QueryBuilder& qb,
                            bool s1, bool s2, bool s3, bool s4, bool s5) {
    auto supplier = qb.scan("supplier", {"s_nationkey", "s_suppkey"});
    auto customer = qb.scan("customer", {"c_custkey", "c_nationkey"});
    auto orders   = qb.scan("orders",   {"o_custkey", "o_orderkey"});
    auto lineitem = qb.scan("lineitem",
        {"l_orderkey", "l_suppkey", "l_shipdate",
         "l_extendedprice", "l_discount"});

    // Two independent {FRANCE, GERMANY} nation instances, columns
    // renamed to n1_*/n2_* via a pure-ColRef project (rename emulation).
    auto make_nation = [&](const char* key_name, const char* name_name) {
        auto n = qb.scan("nation", {"n_nationkey", "n_name"});
        auto nf = n.filter(string_op(StringOpKind::In,
                                     n.c("n_name"), {"FRANCE", "GERMANY"}));
        return nf.project({
            {key_name,  nf.c("n_nationkey")},
            {name_name, nf.c("n_name")},
        });
    };
    auto n1 = make_nation("n1_nationkey", "n1_name");
    auto n2 = make_nation("n2_nationkey", "n2_name");

    // join(caller, arg, on, swap): swap=false → caller is build (left),
    // matching the source's `caller.inner_join(arg)`; swap=true builds
    // on arg.
    auto join = [](const PlanBuilder& caller, const PlanBuilder& arg,
                   Expr on, bool swap) {
        return swap ? arg.inner_join(caller, std::move(on))
                    : caller.inner_join(arg, std::move(on));
    };

    // sn = n1 ⋈ supplier → {s_suppkey, n1_name}  (supplier side)
    auto sn_join = join(n1, supplier,
        n1.c("n1_nationkey") == supplier.c("s_nationkey"), s1);
    auto sn = sn_join.project({
        {"s_suppkey", supplier.c("s_suppkey")},
        {"n1_name",   n1.c("n1_name")},
    });

    // cn = n2 ⋈ customer → {c_custkey, n2_name}  (customer side)
    auto cn_join = join(n2, customer,
        n2.c("n2_nationkey") == customer.c("c_nationkey"), s2);
    auto cn = cn_join.project({
        {"c_custkey", customer.c("c_custkey")},
        {"n2_name",   n2.c("n2_name")},
    });

    // cno = orders ⋈ cn → {o_orderkey, n2_name}
    auto cno_join = join(orders, cn,
        orders.c("o_custkey") == cn.c("c_custkey"), s3);
    auto cno = cno_join.project({
        {"o_orderkey", orders.c("o_orderkey")},
        {"n2_name",    cn.c("n2_name")},
    });

    // cnol = lineitem(date-filtered) ⋈ cno
    //        → {l_suppkey, l_shipdate, l_extendedprice, l_discount, n2_name}
    auto l_filt = lineitem.filter(
            (lineitem.c("l_shipdate") >= lit(19950101))
         && (lineitem.c("l_shipdate") <= lit(19961231)));
    auto cnol_join = join(l_filt, cno,
        l_filt.c("l_orderkey") == cno.c("o_orderkey"), s4);
    auto cnol = cnol_join.project({
        {"l_suppkey",       l_filt.c("l_suppkey")},
        {"l_shipdate",      l_filt.c("l_shipdate")},
        {"l_extendedprice", l_filt.c("l_extendedprice")},
        {"l_discount",      l_filt.c("l_discount")},
        {"n2_name",         cno.c("n2_name")},
    });

    // final = cnol ⋈ sn, then keep only the two valid nation pairs.
    auto final_join = join(cnol, sn,
        cnol.c("l_suppkey") == sn.c("s_suppkey"), s5);
    auto name_filter =
           (string_op(StringOpKind::Equal, sn.c("n1_name"), {"FRANCE"})
             && string_op(StringOpKind::Equal, cnol.c("n2_name"), {"GERMANY"}))
        || (string_op(StringOpKind::Equal, sn.c("n1_name"), {"GERMANY"})
             && string_op(StringOpKind::Equal, cnol.c("n2_name"), {"FRANCE"}));
    auto filtered = final_join.filter(std::move(name_filter));

    auto projected = filtered.project({
        {"supp_nation", sn.c("n1_name")},
        {"cust_nation", cnol.c("n2_name")},
        {"l_year",      year_of(cnol.c("l_shipdate"))},
        {"volume",
            cnol.c("l_extendedprice") * (lit(1.0) - cnol.c("l_discount"))},
    });
    auto agg = projected.aggregate(
        {"supp_nation", "cust_nation", "l_year"},
        {{"revenue", AggKind::Sum, projected.c("volume")}});
    return agg.sort({
        asc("supp_nation"), asc("cust_nation"), asc("l_year"),
    }).plan();
}

PlanPtr build_q71(const QueryBuilder& qb) { return build_q7_alt(qb, false,false,false,false,false); }
PlanPtr build_q72(const QueryBuilder& qb) { return build_q7_alt(qb, true, true, true, true, true ); }
PlanPtr build_q73(const QueryBuilder& qb) { return build_q7_alt(qb, false,false,true, true, true ); }

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

// ── Q91–Q92: alternative Q9 plan (ported from an external IR) ───────
//
// A different join order from build_q9 above: it starts from the green
// partsupp slice (partsupp ⋈ green part), attaches supplier ⋈ nation,
// then joins lineitem (multi-key) and orders, projecting aggressively
// after each join. build_q9 instead starts from part ⋈ lineitem and
// joins nation last.
//
// gamax mapping notes:
//   * `like("p_name", "%green%")` → StringOpKind::Contains.
//   * The lineitem ⋈ pspsn join is multi-key. It is an Inner join, so
//     the G1 rule splits it into single-key + a residual filter (same
//     as build_q9's plsps join). The selective partkey equality is
//     written first so G1 splits on it — see the note at the join.
//   * The five inner joins' build/probe order is parameterized by
//     s1..s5. In gamax `X.inner_join(Y)` puts X on the build (left)
//     side; swap=false keeps the source's caller as build, swap=true
//     builds on the argument.
//
// In this plan the source always writes `large.inner_join(small)` — the
// build-worthy (smaller) relation is the argument of every join — so
// the small-side-build optimum is simply "swap every join":
//   Q91 (00000) every join as-written (source caller = build)
//   Q92 (11111) every join swapped   (source argument = build = the
//               smaller side of all five joins)
static PlanPtr build_q9_alt(const QueryBuilder& qb,
                            bool s1, bool s2, bool s3, bool s4, bool s5) {
    auto part     = qb.scan("part",     {"p_name", "p_partkey"});
    auto partsupp = qb.scan("partsupp",
        {"ps_suppkey", "ps_partkey", "ps_supplycost"});
    auto supplier = qb.scan("supplier", {"s_suppkey", "s_nationkey"});
    auto nation   = qb.scan("nation",   {"n_nationkey", "n_name"});
    auto lineitem = qb.scan("lineitem",
        {"l_suppkey", "l_partkey", "l_orderkey",
         "l_extendedprice", "l_discount", "l_quantity"});
    auto orders   = qb.scan("orders",   {"o_orderkey", "o_orderdate"});

    // join(caller, arg, on, swap): swap=false → caller is build (left),
    // matching the source's `caller.inner_join(arg)`; swap=true builds
    // on arg.
    auto join = [](const PlanBuilder& caller, const PlanBuilder& arg,
                   Expr on, bool swap) {
        return swap ? arg.inner_join(caller, std::move(on))
                    : caller.inner_join(arg, std::move(on));
    };

    // Green parts, projected to just the key.
    auto part_green = part.filter(
        string_op(StringOpKind::Contains, part.c("p_name"), {"green"}));
    auto p = part_green.project({{"p_partkey", part_green.c("p_partkey")}});

    // psp = partsupp restricted to green parts.
    auto psp = join(partsupp, p,
        partsupp.c("ps_partkey") == p.c("p_partkey"), s1);

    // sn = supplier ⋈ nation → {s_suppkey, n_name}.
    auto sn_join = join(supplier, nation,
        supplier.c("s_nationkey") == nation.c("n_nationkey"), s2);
    auto sn = sn_join.project({
        {"s_suppkey", supplier.c("s_suppkey")},
        {"n_name",    nation.c("n_name")},
    });

    // pspsn = green partsupp carrying its supplier's nation.
    auto pspsn = join(psp, sn,
        psp.c("ps_suppkey") == sn.c("s_suppkey"), s3);

    // lpspsn = lineitem ⋈ pspsn (multi-key: partkey + suppkey). The
    // G1 rule splits a multi-key inner join on its FIRST equi clause
    // and makes the rest a residual filter, so the partkey equality is
    // written first: partkey (restricted to green parts) is selective,
    // whereas splitting on suppkey first materialises a ~2.4B-row
    // intermediate at SF=100 (every lineitem × its supplier's green
    // partsupp rows) and OOMs before the residual can prune it.
    auto lpspsn_join = join(lineitem, pspsn,
            (lineitem.c("l_partkey") == pspsn.c("ps_partkey"))
         && (lineitem.c("l_suppkey") == pspsn.c("ps_suppkey")), s4);
    auto lpspsn = lpspsn_join.project({
        {"l_orderkey",      lineitem.c("l_orderkey")},
        {"l_extendedprice", lineitem.c("l_extendedprice")},
        {"l_discount",      lineitem.c("l_discount")},
        {"l_quantity",      lineitem.c("l_quantity")},
        {"ps_supplycost",   pspsn.c("ps_supplycost")},
        {"n_name",          pspsn.c("n_name")},
    });

    // all = orders ⋈ lpspsn, carrying o_orderdate.
    auto all_join = join(orders, lpspsn,
        orders.c("o_orderkey") == lpspsn.c("l_orderkey"), s5);
    auto all = all_join.project({
        {"l_extendedprice", lpspsn.c("l_extendedprice")},
        {"l_discount",      lpspsn.c("l_discount")},
        {"l_quantity",      lpspsn.c("l_quantity")},
        {"ps_supplycost",   lpspsn.c("ps_supplycost")},
        {"n_name",          lpspsn.c("n_name")},
        {"o_orderdate",     orders.c("o_orderdate")},
    });

    // amount = l_extendedprice*(1 - l_discount) - ps_supplycost*l_quantity
    auto projected = all.project({
        {"nation", all.c("n_name")},
        {"o_year", year_of(all.c("o_orderdate"))},
        {"amount",
            all.c("l_extendedprice") * (lit(1.0) - all.c("l_discount"))
          - all.c("ps_supplycost") * all.c("l_quantity")},
    });
    auto agg = projected.aggregate(
        {"nation", "o_year"},
        {{"sum_profit", AggKind::Sum, projected.c("amount")}});
    return agg.sort({asc("nation"), desc("o_year")}).plan();
}

PlanPtr build_q91(const QueryBuilder& qb) { return build_q9_alt(qb, false,false,false,false,false); }
PlanPtr build_q92(const QueryBuilder& qb) { return build_q9_alt(qb, true, true, true, true, true ); }

// ── Q10: Returned Item Reporting ───────────────────────────────────
//
// Returns the top-20 customers (by revenue from returned lineitems) in
// 1993Q4. The aggregate is pushed below the customer/nation joins —
// revenue per o_custkey on (lineitem_filt ⋈ orders_filt) — and then the
// surviving custkeys are enriched via customer (PK c_custkey) and
// nation (PK n_nationkey). Both join legs are inner equi-joins against
// PK / FK-constrained columns (TPC-H guarantees every o_custkey is in
// customer.c_custkey and every c_nationkey is in nation.n_nationkey),
// so each surviving custkey gets exactly one customer row and one
// nation row — no row-drop, no row-multiplication, per-custkey revenue
// preserved bit-for-bit.
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

    auto lo = l_filt.inner_join(o_filt,
        l_filt.c("l_orderkey") == o_filt.c("o_orderkey"));
    auto lo_proj = lo.project({
        {"o_custkey",  lo.c("o_custkey")},
        {"disc_price", lo.c("l_extendedprice") * (lit(1.0) - lo.c("l_discount"))},
    });
    auto lo_agg = lo_proj.aggregate(
        {"o_custkey"},
        {{"revenue", AggKind::Sum, lo_proj.c("disc_price")}});

    // Same join order as base Q10: nation as build (25 rows) avoids
    // partition_hash_join bucket overflow. Customer joins lo_agg on
    // c_custkey == o_custkey (each surviving o_custkey hits exactly
    // one customer row); nation joins via c_nationkey == n_nationkey.
    auto loc = c.inner_join(lo_agg,
        c.c("c_custkey") == lo_agg.c("o_custkey"));
    auto locn = n.inner_join(loc,
        n.c("n_nationkey") == loc.c("c_nationkey"));

    auto out = locn.project({
        {"c_custkey", locn.c("c_custkey")},
        {"c_name",    locn.c("c_name")},
        {"revenue",   locn.c("revenue")},
        {"c_acctbal", locn.c("c_acctbal")},
        {"n_name",    locn.c("n_name")},
        {"c_address", locn.c("c_address")},
        {"c_phone",   locn.c("c_phone")},
        {"c_comment", locn.c("c_comment")},
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
//
// The supplier⋈nation_filt join puts n_filt (1 row after the GERMANY
// filter) on the build side so HASH+GFUR builds a 1-bucket table and
// streams supplier through it.
PlanPtr build_q11(const QueryBuilder& qb) {
    auto ps = qb.scan("partsupp",
        {"ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost"});
    auto s  = qb.scan("supplier", {"s_suppkey", "s_nationkey"});
    auto n  = qb.scan("nation",   {"n_nationkey", "n_name"});

    auto n_filt = n.filter(
        string_op(StringOpKind::Equal, n.c("n_name"), {"GERMANY"}));
    auto sn = n_filt.inner_join(s,
        n_filt.c("n_nationkey") == s.c("s_nationkey"));
    auto pssn = ps.inner_join(sn,
        ps.c("ps_suppkey") == sn.c("s_suppkey"));

    auto with_value = pssn.project({
        {"ps_partkey", pssn.c("ps_partkey")},
        {"value", pssn.c("ps_supplycost") * pssn.c("ps_availqty")},
    });

    auto grouped = with_value.aggregate(
        {"ps_partkey"},
        {{"partkey_value", AggKind::Sum, with_value.c("value")}});

    auto threshold_agg = with_value.aggregate(
        /*keys=*/{},
        {{"global_sum", AggKind::Sum, with_value.c("value")}});
    auto threshold = threshold_agg.project({
        {"threshold", threshold_agg.c("global_sum") * lit(0.0001)},
    });

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
// Correlated subquery: per-part average quantity threshold, joined
// back to the brand=Brand#23 / container=MED BOX slice on l_partkey.
// The threshold is derived from `pl = part_filt ⋈ lineitem` directly
// — pl is an inner join with no extra predicate, so for every
// surviving partkey it contains all lineitem rows for that partkey,
// and per-partkey avg(l_quantity) over pl equals the per-partkey avg
// the spec subquery computes. The avoided second lineitem scan and
// full-table groupby is the bulk of the cost at large SF.
//
// `threshold.inner_join(pl, ...)` keeps the smaller per-partkey side
// on the build side of HASH+GFUR.
PlanPtr build_q17(const QueryBuilder& qb) {
    auto p = qb.scan("part", {"p_partkey", "p_brand", "p_container"});
    auto l = qb.scan("lineitem",
        {"l_orderkey", "l_partkey", "l_quantity", "l_extendedprice"});
    auto p_filt = p.filter(
            string_op(StringOpKind::Equal, p.c("p_brand"), {"Brand#23"})
         && string_op(StringOpKind::Equal, p.c("p_container"), {"MED BOX"}));
    auto pl = p_filt.inner_join(l,
        p_filt.c("p_partkey") == l.c("l_partkey"));

    auto per_pl = pl.aggregate(
        {"l_partkey"},
        {{"avg_qty", AggKind::Avg, pl.c("l_quantity")}});
    auto threshold = per_pl.project({
        {"l_partkey", per_pl.c("l_partkey")},
        {"thr", lit(0.2) * per_pl.c("avg_qty")},
    });

    auto with_thresh = threshold.inner_join(pl,
        threshold.c("l_partkey") == pl.c("l_partkey"));
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
// The IN-subquery "orders whose lineitem sum(quantity) > 300" is
// lowered as a single per-orderkey aggregate over lineitem; the big
// orders' sum_qty is carried through the joins as a payload, so we
// don't need a second lineitem scan or a 5-key outer groupby. Each
// inner-join leg is 1:1 (orders.o_orderkey is PK; customer.c_custkey
// is PK with FK-constrained o_custkey), so the per-orderkey sum
// reaches the output untouched.
PlanPtr build_q18(const QueryBuilder& qb) {
    auto l = qb.scan("lineitem", {"l_orderkey", "l_quantity"});
    auto sum_per_order = l.aggregate(
        {"l_orderkey"},
        {{"sum_qty", AggKind::Sum, l.c("l_quantity")}});
    auto big = sum_per_order.filter(
        sum_per_order.c("sum_qty") > lit(300.0));

    auto o = qb.scan("orders",
        {"o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"});
    auto c = qb.scan("customer", {"c_custkey", "c_name"});

    auto ol  = o.inner_join(big,
        o.c("o_orderkey") == big.c("l_orderkey"));
    auto loc = ol.inner_join(c,
        ol.c("o_custkey") == c.c("c_custkey"));

    auto out = loc.project({
        {"c_name",        loc.c("c_name")},
        {"c_custkey",     loc.c("c_custkey")},
        {"o_orderkey",    loc.c("o_orderkey")},
        {"o_orderdate",   loc.c("o_orderdate")},
        {"o_totalprice",  loc.c("o_totalprice")},
        {"sum_quantity",  loc.c("sum_qty")},
    });
    return out.sort({desc("o_totalprice"), asc("o_orderdate")})
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
// Deep nested IN / correlated threshold: select CANADA suppliers whose
// partsupp entries for "forest" parts have availqty > 0.5 · sum of the
// shipdate-bounded lineitem quantity for that (part, supplier).
//
// A semi-join-reduction plan: rather than aggregating ALL date-filtered
// lineitem and pruning afterwards, it first restricts partsupp to
// (forest parts × CANADA suppliers) — `partsupp_forest_ca` — and reuses
// that small relation to (a) prune lineitem before the
// per-(partkey,suppkey) sum and (b) drive the availqty>threshold check.
// Avoiding the full-lineitem group-by is ~2.4× faster than the naive
// aggregate-everything-then-prune shape at SF=100.
//
// Two gamax-specific notes:
//
//  * Build/probe order: each inner join puts the smaller relation on
//    the build (left) side — `nation_ca` is a single row after the
//    CANADA filter, so it builds and streams supplier through it.
//    Putting the large side as build (the naive form) was ~2× slower.
//
//  * gamax has no multi-key SEMI join — the G1/G7 split rules in
//    rules.cpp fire only on Inner joins, so a multi-key semi-join hits
//    `not_implemented`. Pruning lineitem by (l_partkey, l_suppkey) ∈
//    partsupp_forest_ca is therefore expressed as an INNER join, which
//    is exactly equivalent here: partsupp's (ps_partkey, ps_suppkey) is
//    a primary key, so partsupp_forest_ca has unique join keys and the
//    inner join neither drops nor multiplies lineitem rows; the
//    following group-by consumes only l_* columns. G1 splits the
//    multi-key inner join into single-key + a residual filter.
//
// `supplier_ca` and `partsupp_forest_ca` are each consumed twice, so
// the IR is a DAG (shared shared_ptr sub-plans); lowering memoizes by
// node identity and emits each shared subtree exactly once. See the
// "Reusable sub-DAGs" note in docs/TPCH_JOURNEY.md.
PlanPtr build_q20(const QueryBuilder& qb) {
    auto supplier = qb.scan("supplier",
        {"s_suppkey", "s_name", "s_address", "s_nationkey"});
    auto nation   = qb.scan("nation",   {"n_nationkey", "n_name"});
    auto part     = qb.scan("part",     {"p_partkey", "p_name"});
    auto partsupp = qb.scan("partsupp",
        {"ps_partkey", "ps_suppkey", "ps_availqty"});
    auto lineitem = qb.scan("lineitem",
        {"l_partkey", "l_suppkey", "l_quantity", "l_shipdate"});

    // supplier_ca = suppliers in CANADA, projected to {suppkey, name,
    // address}. nation_ca (1 row) is the build side. Reused twice below.
    auto nation_ca = nation.filter(
        string_op(StringOpKind::Equal, nation.c("n_name"), {"CANADA"}));
    auto sca_join = nation_ca.inner_join(supplier,
        supplier.c("s_nationkey") == nation_ca.c("n_nationkey"));
    auto supplier_ca = sca_join.project({
        {"s_suppkey", supplier.c("s_suppkey")},
        {"s_name",    supplier.c("s_name")},
        {"s_address", supplier.c("s_address")},
    });

    // part_forest = parts whose name starts with 'forest'.
    auto part_forest = part.filter(
        string_op(StringOpKind::Startswith, part.c("p_name"), {"forest"}));

    // partsupp_forest_ca = partsupp restricted to CANADA suppliers AND
    // forest parts (two single-key semi-joins). Reused twice (DAG).
    auto psfca1 = partsupp.left_semi_join(supplier_ca,
        partsupp.c("ps_suppkey") == supplier_ca.c("s_suppkey"));
    auto partsupp_forest_ca = psfca1.left_semi_join(part_forest,
        psfca1.c("ps_partkey") == part_forest.c("p_partkey"));

    // qty_threshold: per-(partkey,suppkey) 0.5·sum(l_quantity) over the
    // date window, restricted to partsupp_forest_ca pairs via the inner
    // join described in the header. The group-by consumes only l_*
    // columns, so the carried partsupp columns are inert.
    auto l_filt = lineitem.filter(
            (lineitem.c("l_shipdate") >= lit(19940101))
         && (lineitem.c("l_shipdate") <  lit(19950101)));
    auto l_pruned = partsupp_forest_ca.inner_join(l_filt,
            (partsupp_forest_ca.c("ps_partkey") == l_filt.c("l_partkey"))
         && (partsupp_forest_ca.c("ps_suppkey") == l_filt.c("l_suppkey")));
    auto qty = l_pruned.aggregate(
        {"l_partkey", "l_suppkey"},
        {{"sum_quantity", AggKind::Sum, l_filt.c("l_quantity")}});
    auto qty_threshold = qty.project({
        {"l_partkey",     qty.c("l_partkey")},
        {"l_suppkey",     qty.c("l_suppkey")},
        {"qty_threshold", qty.c("sum_quantity") * lit(0.5)},
    });

    // over_threshold: partsupp_forest_ca ⋈ qty_threshold, keep the
    // entries whose availqty exceeds the threshold. qty_threshold (the
    // grouped, one-row-per-pair side) is the build side.
    auto ot_join = qty_threshold.inner_join(partsupp_forest_ca,
            (partsupp_forest_ca.c("ps_partkey") == qty_threshold.c("l_partkey"))
         && (partsupp_forest_ca.c("ps_suppkey") == qty_threshold.c("l_suppkey")));
    auto over_threshold = ot_join.filter(
        partsupp_forest_ca.c("ps_availqty") > qty_threshold.c("qty_threshold"));

    // Final: keep CANADA suppliers that appear in over_threshold.
    auto s_final = supplier_ca.left_semi_join(over_threshold,
        supplier_ca.c("s_suppkey") == over_threshold.c("ps_suppkey"));
    auto projected = s_final.project({
        {"s_name",    supplier_ca.c("s_name")},
        {"s_address", supplier_ca.c("s_address")},
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

// Apply per-query lowering hints that aren't expressible at the IR
// level — currently only known group-cardinality bounds. Reused by all
// three driver consumers (run_tpch_ir, benchmark_tpch_ir, axis_probe)
// so the hints are spelled in exactly one place.
//
// Q1 groups by (l_returnflag, l_linestatus) — the spec restricts both
// to 1-char enums and the lineitem data only ever produces 4 distinct
// pairs. Without this hint, the hash groupby's table is sized at
// `min(input_rows, MAXIMUM_SIZE) = input_rows` (~600M at SF=100),
// blowing the 88 GiB UserSpaceMM slab with std::bad_alloc. With the
// hint, the table is sized for 4 groups — a few bytes.
void apply_query_hints(int qid, gamax::ir::LowerOptions& opts) {
    if (qid == 1) {
        if (!opts.group_by) {
            opts.group_by = gamax::groupby::groupby_hint{};
        }
        opts.group_by->groups_upper_bound = 4;
    }
    // Q21 has three groupbys; the inner two are integer-keyed (orderkey)
    // and benefit from axis-sweep coverage of non-HASH algos, but the
    // outer one is single-col STRING (s_name) and would throw on any
    // sort-/partition-based axis cell. Opt into the silent HASH fallback
    // so the inner two still exercise the configured algo while the outer
    // transparently downgrades to HASH. The flag carries no effect when
    // the configured algo is already HASH (the bench default) or when
    // the input shape is supported.
    if (qid == 21) {
        if (!opts.group_by) {
            opts.group_by = gamax::groupby::groupby_hint{};
        }
        opts.group_by->allow_hash_fallback = true;
        // Note: `opts.distinct` is intentionally not touched here.
        // Distinct's hint (algo + SKF) is now a measurable bench axis;
        // letting `apply_query_hints` override it would defeat the
        // sweep. Non-bench callers (run_tpch_ir, tests) leave
        // `opts.distinct` empty and lowering falls back to
        // `opts.group_by` (default HASH, no SKF).
    }
}

} // namespace gamax::ir::tpch
