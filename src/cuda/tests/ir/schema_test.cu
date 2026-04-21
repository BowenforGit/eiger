#include <gtest/gtest.h>

#include "ir/catalog.hpp"
#include "ir/column_id.hpp"
#include "ir/schema.hpp"

namespace gir = gamax::ir;

TEST(IrColumnId, FreshIdsAreDistinct) {
    gir::IdGenerator gen;
    auto a = gen.next();
    auto b = gen.next();
    auto c = gen.next();
    EXPECT_NE(a, b);
    EXPECT_NE(b, c);
    EXPECT_NE(a, c);
}

TEST(IrColumnId, DefaultIsUnsetSentinel) {
    gir::ColumnId unset;
    EXPECT_EQ(unset.value, -1);
    EXPECT_FALSE(unset.is_valid());
    gir::IdGenerator gen;
    auto fresh = gen.next();
    EXPECT_NE(unset, fresh);
    EXPECT_TRUE(fresh.is_valid());
}

TEST(IrSchema, IndexOfById) {
    gir::IdGenerator gen;
    auto a = gen.next();
    auto b = gen.next();
    gir::Schema s({
        {a, gamax::data_type{gamax::type_id::INT32}, false, "a"},
        {b, gamax::data_type{gamax::type_id::FLOAT64}, true,  "b"},
    });
    EXPECT_EQ(s.index_of(a), std::optional<size_t>{0});
    EXPECT_EQ(s.index_of(b), std::optional<size_t>{1});
    gir::ColumnId absent;
    EXPECT_EQ(s.index_of(absent), std::nullopt);
}

TEST(IrSchema, IndexOfByName) {
    gir::IdGenerator gen;
    gir::Schema s({
        {gen.next(), gamax::data_type{gamax::type_id::INT32}, false, "foo"},
        {gen.next(), gamax::data_type{gamax::type_id::INT64}, false, "bar"},
    });
    EXPECT_EQ(s.index_of(std::string("bar")), std::optional<size_t>{1});
    EXPECT_EQ(s.index_of(std::string("nope")), std::nullopt);
}

TEST(IrSchema, RequireIndexThrowsWhenAbsent) {
    gir::Schema s;
    gir::ColumnId absent;
    absent.value = 99;
    EXPECT_THROW(s.require_index(absent), std::runtime_error);
}

TEST(IrCatalog, AddAndFind) {
    gir::Catalog cat;
    gir::CatalogTable lineitem;
    lineitem.name = "lineitem";
    lineitem.column_names = {"l_orderkey", "l_partkey"};
    lineitem.column_types = {
        gamax::data_type{gamax::type_id::INT64},
        gamax::data_type{gamax::type_id::INT64},
    };
    lineitem.column_nullable = {false, false};
    cat.add_table(lineitem);

    const auto* found = cat.find("lineitem");
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->column_names.size(), 2u);
    EXPECT_EQ(found->column_names[0], "l_orderkey");

    EXPECT_EQ(cat.find("missing"), nullptr);
}

TEST(IrCatalog, DuplicateTableNameThrows) {
    gir::Catalog cat;
    gir::CatalogTable t;
    t.name = "t";
    t.column_names = {"c"};
    t.column_types = {gamax::data_type{gamax::type_id::INT32}};
    t.column_nullable = {false};
    cat.add_table(t);
    EXPECT_THROW(cat.add_table(t), std::runtime_error);
}

TEST(IrCatalog, MismatchedMetadataSizesThrows) {
    gir::Catalog cat;
    gir::CatalogTable t;
    t.name = "t";
    t.column_names = {"a", "b"};
    t.column_types = {gamax::data_type{gamax::type_id::INT32}};  // size 1
    t.column_nullable = {false, false};
    EXPECT_THROW(cat.add_table(t), std::runtime_error);
}

TEST(IrCatalog, EmptyNameThrows) {
    gir::Catalog cat;
    gir::CatalogTable t;  // name is default-empty
    EXPECT_THROW(cat.add_table(t), std::runtime_error);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
