package org.calcite;

/*import static org.junit.jupiter.api.Assertions.assertEquals;*/

import java.net.URL;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import javax.sql.DataSource;

import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.util.SourceStringReader;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.Planner;
import org.apache.calcite.adapter.java.ReflectiveSchema;
import org.apache.calcite.adapter.jdbc.JdbcSchema;
import org.apache.calcite.jdbc.CalciteConnection;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.plan.RelOptUtil;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class QueryPlan {
    private static boolean printedPlan = false;
    /*static Logger logger = LoggerFactory.getLogger(QueryPlan.class);*/
    public static void main( String[] args ) throws Exception {
        Properties info = new Properties();
        info.put("model", getPath("model.json"));
        Connection connection = DriverManager.getConnection("jdbc:calcite:", info);
        CalciteConnection calciteConnection = connection.unwrap(CalciteConnection.class);

        SchemaPlus rootSchema = calciteConnection.getRootSchema();
        /*System.out.println(rootSchema.toString());*/
        final DataSource ds = JdbcSchema.dataSource(
                "jdbc:postgresql://localhost:5432/multidb",
                "org.postgresql.Driver",
                "postgres",
                "postgres");
        rootSchema.add("MULTIDB", JdbcSchema.create(rootSchema, "MULTIDB", ds, null, null));
        /*System.out.println(rootSchema.toString());*/

        FrameworkConfig config = Frameworks.newConfigBuilder()
                .defaultSchema(rootSchema)
                .build();

        Planner planner = Frameworks.getPlanner(config);
        SqlNode sqlNode = planner.parse(new SourceStringReader(args[0]));
        /*System.out.println(sqlNode.toString());*/
        sqlNode = planner.validate(sqlNode);
        RelRoot relRoot = planner.rel(sqlNode);
        /*System.out.println(relRoot.toString());*/
        RelNode relNode = relRoot.project();
        String relNodeString = RelOptUtil.toString(relNode);
        System.out.println(relNodeString);
    }

    private static String getPath(String model) {
        URL url = ClassLoader.getSystemClassLoader().getResource(model);
        /*logger.info("path fetched :" + url.getPath());*/
        return url.getPath();
    }

}









/*
THIS WORKS!!!!!!!!!!!!!!!!!!!!!!!
public class CalciteUnit {
    static Logger logger = LoggerFactory.getLogger(CalciteUnit.class);
    public static void main( String[] args ) {
        Properties info = new Properties();
        info.put("model", getPath("model.json"));

        try (Connection connection = DriverManager.getConnection("jdbc:calcite:", info)) {
            Statement statement = connection.createStatement();
            ResultSet resultSet = statement.executeQuery(args[0]);
            System.out.println("Result set:");
            System.out.println(resultSet.toString());

            List<Integer> tradeIds = new ArrayList<>();
            while (resultSet.next()) {
                tradeIds.add(resultSet.getInt("tradeid"));
            }
            System.out.println("Trade ids:");
            System.out.println(tradeIds.toString());
        } catch (Exception e) {
            System.out.println("catch");
        }
    }

    private static String getPath(String model) {
        URL url = ClassLoader.getSystemClassLoader().getResource(model);
        logger.info("path fetched :" + url.getPath());
        return url.getPath();
    }

}*/
