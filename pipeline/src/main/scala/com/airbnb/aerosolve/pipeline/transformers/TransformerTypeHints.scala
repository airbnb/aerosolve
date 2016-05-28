package com.airbnb.aerosolve.pipeline.transformers

import com.airbnb.aerosolve.pipeline.transformers.continuous.Scaler
import org.json4s.TypeHints

/**
 *
 */
class TransformerTypeHints extends TypeHints {

  // TODO (Brad): Reflection hacks and special case handling.
  lazy val classes: List[Class[_]] = List(
    classOf[Scaler],
    classOf[Pipeline[_, _, _]])
  lazy val hintMap: Map[String, Class[_]] = classes.map(clazz => (clazz.getName, clazz)).toMap
  lazy val clazzMap: Map[Class[_], String] = classes.map(clazz => (clazz, clazz.getName)).toMap

  override val hints: List[Class[_]] = classes

  override def hintFor(clazz: Class[_]): String = clazzMap(clazz)

  override def classFor(hint: String): Option[Class[_]] = hintMap.get(hint)
}
