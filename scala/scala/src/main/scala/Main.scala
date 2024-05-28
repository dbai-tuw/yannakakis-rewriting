import java.sql.{Connection, DriverManager}
import java.util.Properties

import org.apache.calcite.adapter.jdbc.JdbcSchema
import org.apache.calcite.jdbc.CalciteConnection
import org.apache.calcite.plan.RelOptUtil
import org.apache.calcite.plan.hep.HepPlanner
import org.apache.calcite.plan.hep.HepProgramBuilder
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.logical.LogicalAggregate
import org.apache.calcite.rel.logical.LogicalFilter
import org.apache.calcite.rel.logical.LogicalProject
import org.apache.calcite.rel.logical.LogicalTableScan
import org.apache.calcite.rel.rel2sql.RelToSqlConverter
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.rules.FilterJoinRule
import org.apache.calcite.rex._
import org.apache.calcite.sql.dialect.Db2SqlDialect
import org.apache.calcite.sql.fun.SqlStdOperatorTable
import org.apache.calcite.sql.parser.SqlParser
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.tools.Frameworks
import org.apache.calcite.tools.RelBuilder
import org.apache.calcite.tools.RelBuilderFactory

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.jdk.CollectionConverters._
import scala.collection.mutable.HashMap

import upickle.default._
import java.nio.file.{Files, Paths}
import java.io.PrintWriter

//import java.net.URL
//import org.apache.calcite.sql.dialect.PostgresqlSqlDialect
//import org.apache.calcite.config.{CalciteConnectionConfig, CalciteSystemProperty}
//import org.apache.calcite.plan.hep.HepProgram
//import org.apache.calcite.rel.RelRoot
//import org.apache.calcite.schema.SchemaPlus
//import org.apache.calcite.sql.SqlNode
//import org.apache.calcite.sql.parser.SqlParseException
//import org.apache.calcite.sql.parser.impl.SqlParserImpl
//import org.apache.calcite.sql.util.SqlString
//import org.apache.calcite.tools.FrameworkConfig
//import org.apache.calcite.tools.ValidationException
//import org.apache.calcite.rel.core.{Filter, Project, TableScan}
//import org.apache.calcite.rel.metadata.RelMetadataQuery
//import org.apache.calcite.rex.RexNode
//import org.apache.calcite.rex.RexCall
//import org.apache.calcite.plan.TableAccessMap
//import org.apache.calcite.rel.RelVisitor
//import org.apache.calcite.rel.core.JoinRelType
//import org.apache.calcite.rel.core.EquiJoin
//import org.apache.calcite.rel.hint.RelHint
//import org.apache.spark.internal.Logging
//import org.apache.spark.sql.catalyst.dsl.expressions.DslExpression
//import org.apache.spark.sql.catalyst.expressions._
//import org.apache.spark.sql.catalyst.expressions.aggregate._
//import org.apache.spark.sql.catalyst.plans.{Inner, InnerLike, LeftSemi}
//import org.apache.spark.sql.catalyst.plans.logical._
//import org.apache.spark.sql.catalyst.rules.Rule
//import org.apache.spark.sql.catalyst.trees.TreePattern
//import org.apache.spark.sql.types._
//import org.apache.spark.sql.types.DecimalType.DoubleDecimal
//import java.io.File

case class JsonOutput(original_query: String, rewritten_query: List[String], features: String, time: Double)
object JsonOutput {
  implicit val rw: ReadWriter[JsonOutput] = macroRW
}

object QueryPlan {
  def main(args: Array[String]): Unit = {
    // stop the time for the whole program
    val startTime = System.nanoTime()

    Class.forName("org.apache.calcite.jdbc.Driver")
    // use the schema information and file locations specified in model.json
    val info = new Properties
    info.put("model", "model.json")

    // connect to the postgresql database
    val connection = DriverManager.getConnection("jdbc:calcite:", info)
    val calciteConnection = connection.unwrap(classOf[CalciteConnection])
    val rootSchema = calciteConnection.getRootSchema
    val ds = JdbcSchema.dataSource("jdbc:postgresql://localhost:5432/multidb", "org.postgresql.Driver", "postgres", "postgres")
    rootSchema.add("MULTIDB", JdbcSchema.create(rootSchema, "MULTIDB", ds, null, null))

    // build a framework for the schema the query corresponds to
    val subSchema = rootSchema.getSubSchema(args(1))
    val parserConfig = SqlParser.Config.DEFAULT.withCaseSensitive(false)
    val config = Frameworks.newConfigBuilder
      .defaultSchema(subSchema)
      .parserConfig(parserConfig)
      .build

    val planner = Frameworks.getPlanner(config)
    // get and parse the SQL input statement
    val query = args(0)
    val sqlNode = planner.parse(query)
    val validatedSqlNode = planner.validate(sqlNode)

    // get the logical query plan
    val relRoot = planner.rel(validatedSqlNode)
    val relNode = relRoot.project

    // push the filters in the logical query plan down
    val relNodeFiltered = decorrelate(relNode)
    // print the logical query plan as string
    val relNodeString = RelOptUtil.toString(relNodeFiltered)
    println(relNodeString)

    // get the references for all attributes
    val att = mutable.Set[RexInputRef]()
    extractInputRefsRecursive(relNode, att)
    att.toSet
    val attributes: Seq[RexInputRef] = att.toSeq
    println("attributes: " + attributes)

    // get the aggregation attributes
    var aggAttributes: Seq[RexNode] = relNodeFiltered match {
      case aggregate: LogicalAggregate =>
        val input = aggregate.getInput
        val aggCalls = aggregate.getAggCallList.asScala
        val hasNonCountAgg = aggCalls.exists(_.getAggregation.getName != "COUNT")
        if (hasNonCountAgg) {
          input match {
            case project: LogicalProject => project.getProjects.asScala.toSeq
            case _ => Seq.empty
          }
        } else { // count(*) case
          Seq.empty
        }
      case _ => // no aggregate case
        Seq.empty
    }
    println("aggAttributes: " + aggAttributes)

    // extract all items and conditions of the joins in the logical plan
    val (items, conditions) = extractInnerJoins(relNodeFiltered)
    //println("items: " + items)
    println("conditions: " + conditions)

    // get the column names for each attribute index
    val names = items.flatMap { i =>
      val fieldNames = i.getRowType().getFieldNames().asScala.toList
      fieldNames
    }
    val indexToName = attributes.zip(names).toMap
    //println("indexToName: " + indexToName)

    // build the hypergraph
    val hg = new Hypergraph(items, conditions, attributes)

    // calculate the join tree
    val jointree = hg.flatGYO

    // there is no jointree, the query is cyclic
    if (jointree == null) {
      println("join is cyclic")
    }
    else {
      // First check if there is a single tree node, i.e., relation that contains all attributes
      // contained in the aggregate functions -> Query 0MA
      if (aggAttributes.isEmpty){
        println("query has no attributes")

        /*
        // Get the output strings for the Bottom Up traversal
        var resultString = ""
        val stringOutput = jointree.BottomUp(indexToName, resultString)
        val stringForJson = stringOutput._2.replace(args(1) + ".", "")
        //val listForJson = stringForJson.split("\n").map(row => s"\"\"\"$row\"\"\"").toList
        val listForJson = stringForJson.split("\n").toList
        val jsonOutput = JsonOutput(args(0), listForJson, "")
        val json: String = write(jsonOutput)
        println(json)
        val filePath = "output.json"
        Files.write(Paths.get(filePath), json.getBytes)
        */
      }
      else {
        val findNodeContainingAttributes = jointree.findNodeContainingAttributes(aggAttributes)
        val nodeContainingAttributes = findNodeContainingAttributes._1
        println("nodeContaining: " + nodeContainingAttributes)
        aggAttributes = findNodeContainingAttributes._2
        println("aggAttofRoot: " + aggAttributes)

        if (nodeContainingAttributes == null) {
          println("query is not 0MA")
        }
        else {
          println("query is 0MA")

          // reroot the tree, such that the root contains all attributes
          val root = nodeContainingAttributes.reroot
          println("new root: " + root + " b: " + root.edges.head.planReference.getRowType)

          // get the aggregate, which are applied at the end on the rerooted root
          val stringAtt = aggAttributes.map{a => indexToName(a.asInstanceOf[RexInputRef])}
          val allAgg: String = relNodeFiltered match {
            case aggregate: LogicalAggregate =>
              val namedAggCalls = aggregate.getNamedAggCalls.asScala
              val zippedResults = namedAggCalls.zip(stringAtt)
              val formattedAgg = zippedResults.map { case (aggCall, att) =>
                val aggStr = aggCall.left.getAggregation
                val name = aggCall.left.name
                s"$aggStr($att) AS $name"
              }
              formattedAgg.mkString(", ")
          }

          // Get the output strings for the Bottom Up traversal
          var resultString = ""
          var dropString = ""
          val stringOutput = root.BottomUp(indexToName, resultString, dropString)
          val stringForJson = stringOutput._2.replace(args(1) + ".", "")
          val listForJson = stringForJson.split("\n").toList

          // add the aggregate to the last CREATE
          val listForJsonLast = listForJson.last
          val modifiedLastString = listForJsonLast.replace("*", allAgg)
          val listForJsonAgg = listForJson.init :+ modifiedLastString

          // write a SELECT of the final table
          val keyword = "TABLE "
          val substringAfterKeyword = listForJsonLast.substring(listForJsonLast.indexOf(keyword) + keyword.length)
          val table = substringAfterKeyword.split("\\s+").headOption.getOrElse("")
          table.trim
          val selectString = "SELECT * FROM " + table

          val finalList = listForJsonAgg ++ List(selectString)
          // val finalList = listForJsonAgg ++ List(selectString) ++ listDrop

          // stop the time for the whole program in seconds and give it to the json
          val endTime = System.nanoTime()
          val executionTime = (endTime - startTime) / 1e9

          /// for the column Date, we needed \\\"Date\\\" for this scala, but now we want Date again
          val original = args(0).replace("\"Date\"", "Date")

          // GET FEATURES OF THE JOIN TREE
          // get the tree depth
          var treeDepth = root.getTreeDepth(root,0)
          println("depth: " + treeDepth)
          // get the item lifetimes
          var containerCounts = root.getContainerCount(hg.getEquivalenceClasses, attributes)
          println("container counts: " + containerCounts)
          // get the branching factor
          var branchingFactors = root.getBranchingFactors(root)
          println("branching factors: " + branchingFactors)
          // get the balancedness factor
          var balancednessFactors = root.getBalancednessFactors(root)
          var balancednessFactor = balancednessFactors._2.sum / balancednessFactors._2.length
          println("balancedness factor: " + balancednessFactors + "  " + balancednessFactor)
          // save all features in one list
          var features = List(treeDepth, containerCounts, branchingFactors, balancednessFactor).toString

          // write a txt file with a visulization of the join tree
          println(root.treeToString(0))
          val filePathJoinTree = "/home/dani/masterarbeit/benchmark_data/benchmark/rewritten/" +
            args(1) + "_" + args(2) + "_jointree.txt"
          val writer = new PrintWriter(filePathJoinTree)
          writer.println(root.treeToString(0))
          writer.close()

          // GET THE HYPERGRAPH REPRESENTATION
          var edgeStart = 0
          val edgeResult = ListBuffer[List[String]]()
          for (i <- items) {
            val edgeCount = i.getRowType().getFieldCount()
            var edgeAtt = attributes.slice(edgeStart,edgeStart + edgeCount)
            val edgeKeys = edgeAtt.map{ e =>
              val keyString = root.edges.head.attributeToVertex.getOrElse(e, e).toString.tail
              keyString
            }
            edgeResult += edgeKeys.toList
            edgeStart = edgeStart + edgeCount
          }
          println("hypergraph representation: " + edgeStart + " " + edgeResult.toString)
          // write a txt file with the edges and the number of vertices of the hypergraph
          val filePathHypergraph = "/home/dani/masterarbeit/benchmark_data/benchmark/rewritten/" +
            args(1) + "_" + args(2) + "_hypergraph.txt"
          val writer1 = new PrintWriter(filePathHypergraph)
          writer1.println(edgeStart + " " + edgeResult.toString)
          writer1.close()


          // write a json file with the original and the rewritten query
          val jsonOutput = JsonOutput(original, finalList, features, executionTime)
          val json: String = write(jsonOutput)
          println(json)
          val filePath = "/home/dani/masterarbeit/benchmark_data/benchmark/rewritten/" +
            args(1) + "_" + args(2) + "_output.json"
          Files.write(Paths.get(filePath), json.getBytes)

          // write a file, which makes dropping the tables after creating them easy
          val listDrop = stringOutput._3.split("\n").toList
          val jsonOutputDrop = JsonOutput("", listDrop, "", 0)
          val jsonDrop: String = write(jsonOutputDrop)
          val filePathDrop = "/home/dani/masterarbeit/benchmark_data/benchmark/rewritten/" +
            args(1) + "_" + args(2) + "_drop.json"
          Files.write(Paths.get(filePathDrop), jsonDrop.getBytes)
        }
      }
    }
  }

  // define the function, which pushes the filters down
  private def decorrelate(root: RelNode): RelNode = {
    val f: RelBuilderFactory = RelBuilder.proto()
    val programBuilder = new HepProgramBuilder
    programBuilder.addRuleInstance(new FilterJoinRule.FilterIntoJoinRule(true, f, FilterJoinRule.TRUE_PREDICATE))
    programBuilder.addRuleInstance(new FilterJoinRule.JoinConditionPushRule(f, FilterJoinRule.TRUE_PREDICATE))
    val program = programBuilder.build
    val planner = new HepPlanner(program)
    planner.setRoot(root)
    planner.findBestExp
  }

  // helper function for extracteInnerJoins, which splits the conjunctive predicates
  def splitConjunctivePredicates(condition: RexNode): Seq[RexNode] = condition match {
    case call: RexCall if call.getKind == SqlKind.AND =>
      val left = call.getOperands.get(0)
      val right = call.getOperands.get(1)
      splitConjunctivePredicates(left) ++ splitConjunctivePredicates(right)
    case predicate if predicate.getKind == SqlKind.EQUALS =>
      Seq(predicate)
    case _ => Seq.empty[RexNode]
  }

  // get the RexInputRefs for all attributes
  private def extractInputRefsRecursive(relNode: RelNode, inputRefs: mutable.Set[RexInputRef]): Unit = {
    val rowType = relNode.getRowType
    val fieldCount = rowType.getFieldCount
    for (i <- 0 until fieldCount) {
      val inputRef = new RexInputRef(i, rowType)
      inputRefs.add(inputRef)
    }
    relNode.getInputs.asScala.foreach { child =>
      extractInputRefsRecursive(child, inputRefs)
    }
  }

  //Extracts items of consecutive inner joins and join conditions
  def extractInnerJoins(plan: RelNode): (Seq[RelNode], Seq[RexNode]) = {
    plan match {
      case join: Join if join.getJoinType == org.apache.calcite.rel.core.JoinRelType.INNER =>
        val left = join.getLeft
        val right = join.getRight
        val cond = join.getCondition
        val (leftPlans, leftConditions) = extractInnerJoins(left)
        val (rightPlans, rightConditions) = extractInnerJoins(right)
        (leftPlans ++ rightPlans, leftConditions ++ rightConditions ++ splitConjunctivePredicates(cond))
      case project: LogicalProject =>
        val input = project.getInput
        val (childPlans, childConditions) = extractInnerJoins(input)
        (childPlans, childConditions)
      case aggregate: LogicalAggregate =>
        val input = aggregate.getInput
        val (childPlans, childConditions) = extractInnerJoins(input)
        (childPlans, childConditions)
      case x =>
        (Seq(plan), Seq.empty[RexNode])
    }
  }

  // define a class for hypergraph edges
  class HGEdge(val vertices: Set[RexNode], var name: String, var nameJoin: String, val planReference: RelNode,
               val attributeToVertex: mutable.Map[RexNode, RexNode], val attIndex: Int,
               val attributes: Seq[RexNode]){

    // get a map between the vertices and attributes
    val vertexToAttribute = HashMap[RexNode, RexNode]()
    val planReferenceAttributes = planReference.getRowType.getFieldList
    planReferenceAttributes.forEach { case att =>
      var index = att.getIndex + attIndex
      var key = attributes(index)
      val keyString = attributeToVertex.getOrElse(key, null)
      val valueString = key
      if (keyString != null) vertexToAttribute.put(keyString, valueString)
    }
    //println("vertexToAttribute: " + vertexToAttribute)

    // check if the vertices of an edge occur in the vertices of another edge
    def contains(other: HGEdge): Boolean = {
      other.vertices subsetOf vertices
    }

    // check if the vertices of two edges are different
    def containsNotEqual(other: HGEdge): Boolean = {
      contains(other) && !(vertices subsetOf other.vertices)
    }

    def copy(newVertices: Set[RexNode] = vertices,
             newName: String = name,
             newPlanReference: RelNode = planReference): HGEdge =
      new HGEdge(newVertices, newName, newName, newPlanReference, attributeToVertex, attIndex, attributes)

    override def toString: String = s"""${name}(${vertices.mkString(", ")})"""
  }

  // define a class for hypertree nodes
  class HTNode(val edges: Set[HGEdge], var children: Set[HTNode], var parent: HTNode){

    // get the SQL statements of the bottom up traversal
    def BottomUp(indexToName: scala.collection.immutable.Map[RexInputRef, String], resultString: String,
                 dropString: String): (RelNode, String, String) = {
      val edge = edges.head
      val scanPlan = edge.planReference
      val vertices = edge.vertices
      var prevJoin: RelNode = scanPlan

      // create the filtered single tables
      val dialect = Db2SqlDialect.DEFAULT
      val relToSqlConverter = new RelToSqlConverter(dialect)
      val res = relToSqlConverter.visitRoot(scanPlan)
      val sqlNode1 = res.asQueryOrValues()
      var result = sqlNode1.toSqlString(dialect, false).getSql()
      result = "CREATE VIEW " + edge.name + " AS " + result
      var resultString1 = resultString + result.replace("\n", " ") + "\n"
      var dropString1 = "DROP VIEW " + edge.name + "\n" + dropString

      // create semi joins with all children
      for (c <- children) {
        val childEdge = c.edges.head
        val childVertices = childEdge.vertices
        val overlappingVertices = vertices.intersect(childVertices)

        val cStringOutput = c.BottomUp(indexToName, resultString1, dropString1)
        resultString1 = cStringOutput._2
        dropString1 = cStringOutput._3
        val newName = edge.nameJoin + childEdge.nameJoin
        var result1 = "CREATE UNLOGGED TABLE " + newName + " AS SELECT * FROM " + edge.nameJoin +
          " WHERE EXISTS (SELECT 1 FROM " + childEdge.nameJoin + " WHERE "
        val joinConditions = overlappingVertices.map { vertex =>
          val att1 = edge.vertexToAttribute(vertex)
          val att2 = childEdge.vertexToAttribute(vertex)
          val att1_name = indexToName.getOrElse(att1.asInstanceOf[RexInputRef], "")
          val att2_name = indexToName.getOrElse(att2.asInstanceOf[RexInputRef], "")
          result1 = result1 + edge.nameJoin + "." + att1_name + "=" + childEdge.nameJoin + "." + att2_name + " AND "
        }
        result1 = result1.dropRight(5) + ")"
        resultString1 = resultString1 + result1 + "\n"
        dropString1 = "DROP TABLE " + newName + "\n" + dropString1
        edge.nameJoin = newName
        childEdge.nameJoin = newName
      }
      (prevJoin, resultString1, dropString1)
    }

    // define a given root as the new root of the tree
    def reroot: HTNode = {
      if (parent == null) {
        this
      } else {
        var current = this
        var newCurrent = this.copy(newParent = null)
        val root = newCurrent
        while (current.parent != null) {
          val p = current.parent
          val newChild = p.copy(newChildren = p.children - current, newParent = null)
          newCurrent.children += newChild
          current = p
          newCurrent = newChild
        }
        root.setParentReferences
        root
      }
    }

    // check if there is a node containing all aggregation attributes (and find it)
    def findNodeContainingAttributes(aggAttributes: Seq[RexNode]): (HTNode, Seq[RexNode]) = {
      var aggAtt = Seq[RexNode]()
      var nodeAtt = List[RexNode]()

      // get the node attributes and the agg attributes, with the same mappings
      val e = edges.head
      val nodeAttributes = e.planReference.getRowType.getFieldList
      nodeAttributes.forEach { case x =>
        var index = x.getIndex + e.attIndex
        var key = e.attributes(index)
        nodeAtt = nodeAtt :+ e.attributeToVertex.getOrElse(key, key)
      }
      aggAtt = aggAttributes.map(key => e.attributeToVertex.getOrElse(key, key))

      // Check if all aggregates are present in this node
      val allInSet = aggAtt.forall(nodeAtt.contains)

      if (allInSet) {
        println("All elements are present in " + this)
        aggAtt = aggAtt.map{a => edges.head.vertexToAttribute.getOrElse(a,a)}
        (this, aggAtt)
      } else {
        for (c <- children) {
          val node = c.findNodeContainingAttributes(aggAttributes)
          if (node != null) {
            return node
          }
        }
        null
      }
    }

    // set references between child-parent relationships in the tree
    def setParentReferences: Unit = {
      for (c <- children) {
        c.parent = this
        c.setParentReferences
      }
    }

    // get the join tree's depth
    def getTreeDepth(root: HTNode, depth: Int): Int = {
      if (root.children.isEmpty) {
        depth
      } else {
        root.children.map(c => getTreeDepth(c, depth + 1)).max
      }
    }

    // get a list of the item lifetimes of all attributes in the join tree
    def getContainerCount(equivalenceClasses: Set[Set[RexNode]], attributes: Seq[RexNode]): List[Int] = {
      // number of the items, which occure several times, are those being joined on
      // how often they appear can be retrived of the size of their equivalence class
      var containerCount = equivalenceClasses.map(_.size).toList
      // the number of attributes only occuring once, are the number of all attribute minus
        // the attributes occuring more often
      val occuringOnce = attributes.size - containerCount.sum
      val occuringOnceList = List.fill(occuringOnce)(1)
      containerCount = occuringOnceList ::: containerCount
      containerCount.sorted
    }

    // get the branching factors of the join tree
    def getBranchingFactors(root: HTNode): List[Int] = {
      if (root.children.isEmpty) {
        List.empty[Int]
      } else {
        var sizes = List.empty[Int]
        for (child <- root.children) {
          sizes = sizes ++ getBranchingFactors(child)
        }
        sizes ::: List(root.children.size)
      }
    }

    // get the balancedness factor of the join tree
    def getBalancednessFactors(root: HTNode): (Int, List[Double]) = {
      if (root.children.isEmpty){
        return (0, List.empty[Double])
      } else if (root.children.size == 1){
        val balanceOneChild = getBalancednessFactors(root.children.head)
        return (balanceOneChild._1, balanceOneChild._2)
      } else {
        val childrenResults = root.children.toList.map(c => getBalancednessFactors(c))
        val firstElements = childrenResults.map(_._1).map(_ + 1)
        val secondElements = childrenResults.map(_._2)
        val combinedSecondElements = secondElements.flatten
        val elementsCount = firstElements.sum
        val balancedness = firstElements.min.toDouble / firstElements.max
        return (elementsCount, combinedSecondElements ::: List(balancedness))
      }
    }

    def copy(newEdges: Set[HGEdge] = edges, newChildren: Set[HTNode] = children,
             newParent: HTNode = parent): HTNode =
      new HTNode(newEdges, newChildren, newParent)

    // define a function to be able to print the join tree
    def treeToString(level: Int = 0): String = {
      s"""${"-- ".repeat(level)}TreeNode(${edges})""" +
        s"""[${edges.map {
          case e if e.planReference.isInstanceOf[LogicalTableScan] =>
            e.planReference.getTable.getQualifiedName
          case e if e.planReference.isInstanceOf[LogicalFilter] =>
            e.planReference.getInputs.get(0).getTable.getQualifiedName
        }}] [[parent: ${parent != null}]]
           |${children.map(c => c.treeToString(level + 1)).mkString("\n")}""".stripMargin
    }

    //override def toString: String = toString(0)
  }

  // define a hypergraph class
  class Hypergraph(private val items: Seq[RelNode], private val conditions: Seq[RexNode],
                   private val attributes: Seq[RexNode]){
                   //private val att_map: HashMap[String, String]) {
    private val vertices: mutable.Set[RexNode] = mutable.Set.empty
    private val edges: mutable.Set[HGEdge] = mutable.Set.empty
    private val attributeToVertex: mutable.Map[RexNode, RexNode] = mutable.Map.empty
    private var equivalenceClasses: Set[Set[RexNode]] = Set.empty

    // add all equality conditions to the equivalence classes
    for (cond <- conditions) {
      cond match {
        case call: RexCall if call.getOperator == SqlStdOperatorTable.EQUALS =>
          val operands = call.getOperands.asScala.toList
          if (operands.size == 2) {
            val lAtt = operands.head
            val rAtt = operands(1)
            equivalenceClasses += Set(lAtt, rAtt)
          }
        case _ => println("other")
      }
    }

    // combine all pairs with common columns/attributes
    // e.g. (col1, col2),(col1, col3),(col4,col5) -> (col1, col2, col3),(col4,col5)
    while (combineEquivalenceClasses) {}
    println("combined equivalence classes: " + equivalenceClasses)

    // design an efficient mapping between the attributes and vertices
    // e.g. vertexToAttributes: col1 -> (col1,col2,col3); col4 -> (col4,col5)
    //      attributeToVertex: col1->col1, col2->col1, col3->col1, col4->col4, col5->col4
    for (equivalenceClass <- equivalenceClasses) {
      val attName = equivalenceClass.head
      vertices.add(attName)
      for (equivAtt <- equivalenceClass) {
        attributeToVertex.put(equivAtt, attName)
      }
    }
    println("attribute to vertex mapping: " + attributeToVertex)

    var tableIndex = 1
    var attIndex = 0
    // iterate over all subtrees
    for (item <- items) {
      //println("join item: " + item)

      // get the attributes for this subtree
      // check if it is in the attributeToVertex list and get consistent naming
      val projectAttributes = item.getRowType.getFieldList
      //println("projectAttributes: " + projectAttributes)

      var projectAtt = List[RexNode]()
      projectAttributes.forEach { case x =>
        var index = x.getIndex + attIndex
        var key = attributes(index)
        projectAtt = projectAtt :+ attributeToVertex.getOrElse(key, null)
        //projectAtt = projectAtt :+ attributeToVertex.getOrElse(key, key)
      }

      // get the hyperedges (were join partners have the same name now)
      val hyperedgeVertices = projectAtt.filter(_ != null).toSet
      val hyperedge = new HGEdge(hyperedgeVertices, s"E${tableIndex}", s"E${tableIndex}", item, attributeToVertex, attIndex, attributes)
      //println("hyperedge: " + hyperedge)
      tableIndex += 1
      attIndex += projectAttributes.size
      //println("he: " + hyperedge + hyperedge.planReference.getTable)
      edges.add(hyperedge)
    }
    println("hyperedges: " + edges)

    // helper function to combine all pairs with common columns
    private def combineEquivalenceClasses: Boolean = {
      for (set <- equivalenceClasses) {
        for (otherSet <- equivalenceClasses - set) {
          if ((set intersect otherSet).nonEmpty) {
            val unionSet = (set union otherSet)
            equivalenceClasses -= set
            equivalenceClasses -= otherSet
            equivalenceClasses += unionSet
            return true
          }
        }
      }
      false
    }

    // get the equivalence classes
    def getEquivalenceClasses: Set[Set[RexNode]] = equivalenceClasses

    // check if the query is acyclic (<=> having a join tree)
    def isAcyclic: Boolean = {
      flatGYO == null
    }

    // compute the join tree
    def flatGYO: HTNode = {
      var gyoEdges: mutable.Set[HGEdge] = mutable.Set.empty
      var mapping: mutable.Map[String, HGEdge] = mutable.Map.empty
      var root: HTNode = null
      var treeNodes: mutable.Map[String, HTNode] = mutable.Map.empty

      for (edge <- edges) {
        mapping.put(edge.name, edge)
        gyoEdges.add(edge.copy())
      }
      //println("mapping: " + mapping)
      //println("GYO edges: " + gyoEdges)

      var progress = true
      while (gyoEdges.size > 1 && progress) {
        for (e <- gyoEdges) {
          val allOtherVertices = (gyoEdges - e).map(o => o.vertices)
            .reduce((o1, o2) => o1 union o2)
          val singleNodeVertices = e.vertices -- allOtherVertices

          val eNew = e.copy(newVertices = e.vertices -- singleNodeVertices)
          gyoEdges = (gyoEdges - e) + eNew

        }

        var nodeAdded = false
        for (e <- gyoEdges) {
          val supersets = gyoEdges.filter(o => o containsNotEqual e)

          if (supersets.isEmpty) {
            val containedEdges = gyoEdges.filter(o => (e contains o) && (e.name != o.name))
            val parentNode = treeNodes.getOrElse(e.name, new HTNode(Set(e), Set(), null))
            val childNodes = containedEdges
              .map(c => treeNodes.getOrElse(c.name, new HTNode(Set(c), Set(), null)))
              .toSet

            parentNode.children ++= childNodes
            if (childNodes.nonEmpty) {
              nodeAdded = true
            }

            treeNodes.put(e.name, parentNode)
            childNodes.foreach(c => treeNodes.put(c.edges.head.name, c))
            root = parentNode
            root.setParentReferences
            gyoEdges --= containedEdges
          }
        }
        if (!nodeAdded) progress = false
      }

      if (gyoEdges.size > 1) {
        return null
      }
      root
    }
  }

}